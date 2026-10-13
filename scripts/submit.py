"""Preview or submit an immutable source snapshot to Beaker. No GPU for CPU stages."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from pipeline import REPO, STAGES, MODEL_STAGES, add_stage_arguments, validate_args

# Existing ai2/flex2 image, verified by Beaker image lookup on 2026-10-12.
GPU_IMAGE = '01KAXVQ0YN242MQ9YEQDQQK512'  # tylerr/olmo-core-tch280cu128-2025-11-25
CPU_IMAGE = 'python:3.12-slim-bookworm'
WEKA_CLUSTERS_PLACEHOLDER = 'ACCESSIBLE_WEKA_CLUSTERS_RESOLVED_AT_SUBMISSION'
RESOURCES = {
    'download-data': (4, '32 GiB'),
    'train-tokenizer': (64, '128 GiB'),
    'tokenize-data': (16, '64 GiB'),
    'train-model': (32, '256 GiB'),
    'convert-model': (8, '64 GiB'),
    'evaluate-model': (8, '64 GiB'),
}


def gpu_count(args):
    return args.gpus if args.stage == 'train-model' else int(args.stage == 'evaluate-model')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    for stage in STAGES:
        s = sub.add_parser(stage)
        add_stage_arguments(s, stage)
        s.add_argument('--workspace', default=os.environ.get('BEAKER_WORKSPACE', 'ai2/flex2'))
        default_cluster = (None if stage == 'download-data' else
                           'ai2/jupiter' if stage in ('train-model', 'evaluate-model') else 'ai2/phobos')
        s.add_argument('--cluster', default=os.environ.get('BEAKER_CLUSTER', default_cluster),
                       help='Optional placement restriction; download defaults to all eligible Weka clusters')
        s.add_argument('--priority', choices=('low', 'normal', 'high', 'urgent'), default='normal')
        s.add_argument('--cpus', type=int, default=RESOURCES[stage][0])
        s.add_argument('--memory', default=RESOURCES[stage][1])
        s.add_argument('--preemptible', action='store_true')
        s.add_argument('--name', help='Beaker experiment name; otherwise generated uniquely')
        s.add_argument('--image', default=GPU_IMAGE if stage in MODEL_STAGES else CPU_IMAGE,
                       help='Beaker image ID for training; Docker image for CPU stages')
        s.add_argument('--hf-token-secret', help='Optional Beaker secret name to expose as HF_TOKEN')
        s.add_argument('--dry-run', action='store_true', help='Print JSON; no upload or job submission')
        s.add_argument('--spec-file', type=Path, help='Also save the generated JSON spec')
        s.add_argument('--receipt', type=Path, help='Save the submitted workload IDs as JSON')
    return p


def stage_arguments(args):
    result = [args.stage, '--root', str(args.root)]
    if args.stage != 'download-data':
        result += ['--vocab-size', str(args.vocab_size), '--data-size', args.data_size]
    if args.stage in MODEL_STAGES:
        for name in ('model_size', 'gpus', 'micro_batch_size', 'batch_size',
                     'learning_rate', 'weight_decay', 'epochs', 'seed', 'max_steps'):
            value = getattr(args, name)
            if value is not None:
                result += ['--' + name.replace('_', '-'), str(value)]
        if args.no_compile:
            result.append('--no-compile')
    if args.stage == 'evaluate-model':
        result += ['--evals', *args.evals]
    return result


def discover_weka_clusters(beaker, workspace):
    """Beaker requires explicit constraints for Weka; discover them from live access metadata."""
    from google.protobuf.json_format import MessageToDict
    user_id = beaker.user.get().id
    budget_id = MessageToDict(workspace).get('budgetId')
    eligible = []
    for cluster in beaker.cluster.list(org=workspace.owner_org):
        info = MessageToDict(cluster)
        if 'storage:weka' not in info.get('tags', []) or not info.get('nodeCount', 0):
            continue
        if info.get('maxTaskTimeout') == '0s':
            continue  # session-only cluster; it rejects batch tasks
        if info.get('userRestrictions') and user_id not in info['userRestrictions']:
            continue
        if info.get('budgetRestrictions') and budget_id not in info['budgetRestrictions']:
            continue
        eligible.append(f'{cluster.organization_name}/{cluster.name}')
    if not eligible:
        raise ValueError('No accessible Weka clusters found for this account/workspace')
    return sorted(eligible)


def make_spec(args, code_dataset='COMMITTED_CODE_DATASET', revision='HEAD', resolved_clusters=None):
    requested_gpus = gpu_count(args)
    gpu_job = requested_gpus > 0
    env = {
        'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'true',
        'RAYON_NUM_THREADS': str(args.cpus),
        'OMP_NUM_THREADS': str(max(1, args.cpus // (requested_gpus or 1))),
        'HF_HOME': str(args.root / '.cache/huggingface'),
        'TOKENIZER_CODE_REVISION': revision,
    }
    if not gpu_job:
        env['CUDA_VISIBLE_DEVICES'] = ''
    env_vars = [{'name': k, 'value': v} for k, v in env.items()]
    if args.hf_token_secret:
        env_vars.append({'name': 'HF_TOKEN', 'secret': args.hf_token_secret})
    context = {'priority': args.priority, 'preemptible': args.preemptible}
    if args.stage == 'train-model':
        # Use the current scheduling field, not the deprecated preemptible flag.
        context = {'priority': args.priority, 'minRuntime': '0s'}
    return {
        'version': 'v2',
        'description': f'Tokenizer project: {args.stage}; source {revision}',
        'tasks': [{
            'name': 'main',
            'image': {'beaker' if args.stage in MODEL_STAGES else 'docker': args.image},
            'command': ['bash', '/code/bootstrap.sh'],
            'arguments': stage_arguments(args),
            'envVars': env_vars,
            'datasets': [
                {'mountPath': '/code', 'source': {'beaker': code_dataset}},
                {'mountPath': '/oe-training-default', 'source': {'weka': 'oe-training-default'}},
            ],
            'result': {'path': '/results'},
            'resources': {'cpuCount': args.cpus, 'memory': args.memory,
                          'gpuCount': requested_gpus,
                          'sharedMemory': '8 GiB' if gpu_job else '1 GiB'},
            # Offline previews use a named placeholder; real submissions resolve
            # every accessible Weka cluster rather than pinning a single cluster.
            'constraints': {'cluster': resolved_clusters or [args.cluster or WEKA_CLUSTERS_PLACEHOLDER]},
            'context': context,
        }],
    }


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def parse_spec(spec):
    """Preserve explicit minRuntime with the installed beaker-py 2.5 client."""
    from dataclasses import dataclass
    from beaker import BeakerExperimentSpec, BeakerTaskContext

    @dataclass
    class RuntimeContext(BeakerTaskContext):
        min_runtime: str | None = None

    compatible = copy.deepcopy(spec)
    runtimes = [task['context'].pop('minRuntime', None) for task in compatible['tasks']]
    parsed = BeakerExperimentSpec.from_json(compatible)
    for task, runtime in zip(parsed.tasks, runtimes):
        if runtime is not None:
            task.context = RuntimeContext(**vars(task.context), min_runtime=runtime)
    return parsed


def save_spec(spec, destination):
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(spec, indent=2) + '\n')


def create_experiment(beaker, args, workspace, name, spec):
    from beaker.exceptions import BeakerError
    import re
    while True:
        save_spec(spec, args.spec_file)
        try:
            return beaker.experiment.create(name=name, workspace=workspace,
                                           spec=parse_spec(spec))
        except BeakerError as error:
            # Budget membership alone does not imply an allocation at every
            # level. A 422 rejects the whole submission before creating a job.
            # Remove only the cluster explicitly identified as inaccessible to
            # this non-preemptible workload; never retry ambiguous outcomes here.
            rejected = (re.search(r'\[code=422\].*cluster "([^"]+)" requires an allocation '
                                  r'for workloads with a minimum runtime', str(error)) or
                        re.fullmatch(r'task "[^"]+": cluster "([^"]+)" doesn\'t allow tasks', str(error)))
            clusters = spec['tasks'][0]['constraints']['cluster']
            if args.stage != 'download-data' or args.cluster or not rejected or rejected[1] not in clusters:
                raise
            clusters.remove(rejected[1])
            print(f'Excluded {rejected[1]}: {error}', flush=True)
            if not clusters:
                raise ValueError('No Weka clusters permit this workload with its current allocation') from error


def main():
    p = parser()
    args = p.parse_args()
    try:
        validate_args(args)
        if args.cpus <= 0:
            raise ValueError('--cpus must be positive')
        args.root.resolve().relative_to('/oe-training-default')
    except ValueError as e:
        p.error(f'{e}; Beaker output must be under the /oe-training-default Weka mount')
    revision = git('rev-parse', 'HEAD')
    spec = make_spec(args, revision=revision)
    if args.dry_run:
        # Intentionally works without a Beaker installation or credentials.
        save_spec(spec, args.spec_file)
        print(json.dumps(spec, indent=2))
        return
    from beaker import Beaker

    if git('status', '--porcelain', '--untracked-files=normal'):
        p.error('Commit all project changes before submission; jobs use git archive HEAD')
    # Parse before any upload/launch. The SDK consumes its input dictionary.
    parse_spec(spec)
    name = args.name or f'tokenizer-{args.stage}-{uuid.uuid4().hex[:12]}'
    with tempfile.TemporaryDirectory(prefix='tokenizer-source-') as temporary:
        folder = Path(temporary)
        archive = folder / 'repo.tar.gz'
        subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar.gz',
                        f'--output={archive}', revision], check=True)
        bootstrap = folder / 'bootstrap.sh'
        bootstrap.write_text(git('show', f'{revision}:scripts/bootstrap.sh') + '\n')
        with Beaker.from_env(default_workspace=args.workspace, check_for_upgrades=False) as b:
            workspace = b.workspace.get(args.workspace)
            cluster = b.cluster.get(args.cluster) if args.cluster else None
            resolved_clusters = [args.cluster] if cluster else discover_weka_clusters(b, workspace)
            hostnames = None
            if args.stage in MODEL_STAGES:
                b.image.get(args.image)
            if gpu_count(args):
                from google.protobuf.json_format import MessageToDict
                hostnames = [n.hostname for n in b.node.list(cluster=cluster)
                             if MessageToDict(n).get('nodeResources', {}).get('gpuType')
                             == 'GPU_TYPE_NVIDIA_H100' and len(n.node_resources.gpu_ids) >= gpu_count(args)]
                if not hostnames:
                    raise RuntimeError('No qualifying H100 nodes in the requested cluster')
            elif args.stage != 'download-data' and cluster.node_shape.gpu_count:
                raise ValueError('CPU stages require a CPU-only cluster to avoid reserving GPU slots')
            dataset = b.dataset.create(
                f'tokenizer-code-{revision[:12]}-{uuid.uuid4().hex[:8]}', archive, bootstrap,
                workspace=workspace, strip_paths=True,
                description=f'shmsw25/tokenizer-vocab-scaling source at {revision}',
            )
            spec = make_spec(args, dataset.id, revision, resolved_clusters)
            if hostnames:
                spec['tasks'][0]['constraints']['hostname'] = hostnames
            workload = create_experiment(b, args, workspace, name, spec)
            if args.receipt:
                from pipeline import write_json
                write_json(args.receipt, {'workload_id': workload.experiment.id,
                           'experiment_id': workload.experiment.id,
                           'url': b.experiment.url(workload), 'code_dataset_id': dataset.id,
                           'revision': revision, 'name': name})
            print(b.experiment.url(workload))
            print(f'Code dataset: {dataset.id} (retain for reproducibility)')
            print('Run the next stage only after this experiment succeeds.')


if __name__ == '__main__':
    main()
