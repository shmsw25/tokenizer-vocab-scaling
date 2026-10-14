"""Preview or submit an immutable source snapshot to Beaker. No GPU for CPU stages."""
import argparse
import copy
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from pipeline import REPO, STAGES, MODEL_STAGES, RAW_STAGES, add_stage_arguments, validate_args
from runtime_batch import GPU_CLUSTERS, GPU_TYPES

# Existing ai2/flex2 image, verified by Beaker image lookup on 2026-10-12.
GPU_IMAGE = '01KAXVQ0YN242MQ9YEQDQQK512'  # tylerr/olmo-core-tch280cu128-2025-11-25
CPU_IMAGE = 'python:3.12-slim-bookworm'
WEKA_CLUSTERS_PLACEHOLDER = 'ACCESSIBLE_WEKA_CLUSTERS_RESOLVED_AT_SUBMISSION'
VALIDATION_CLUSTERS = GPU_CLUSTERS
VALIDATION_GPU_TYPES = GPU_TYPES
RESOURCES = {
    'download-data': (4, '32 GiB'),
    'prepare-validation': (16, '64 GiB'),
    'validate-model': (8, '64 GiB'),
    'sample-pool': (4, '32 GiB'),
    'train-tokenizer': (64, '128 GiB'),
    'tokenize-data': (16, '64 GiB'),
    'tokenize-validation': (16, '64 GiB'),
    'train-model': (32, '256 GiB'),
    'convert-model': (8, '64 GiB'),
    'evaluate-model': (8, '64 GiB'),
}


def gpu_count(args):
    if args.stage == 'train-model':
        return args.gpus
    return int(args.stage in ('evaluate-model', 'validate-model') or getattr(args, 'chain_evaluation', False))


def runtime_nanoseconds(value):
    match = re.fullmatch(r'(\d+)([smh])', value)
    if not match:
        raise ValueError('Minimum runtime must be an integer duration, e.g. 0s or 8h')
    return int(match[1]) * {'s': 1, 'm': 60, 'h': 3600}[match[2]] * 1_000_000_000


def minimum_runtime(value):
    try:
        runtime_nanoseconds(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    for stage in STAGES:
        s = sub.add_parser(stage)
        add_stage_arguments(s, stage)
        s.add_argument('--workspace', default=os.environ.get('BEAKER_WORKSPACE', 'ai2/flex2'))
        default_cluster = (None if stage in RAW_STAGES else
                           list(GPU_CLUSTERS) if stage in ('train-model', 'validate-model') else
                           'ai2/jupiter' if stage == 'evaluate-model' else 'ai2/phobos')
        s.add_argument('--cluster', nargs='+', default=os.environ.get('BEAKER_CLUSTER', default_cluster),
                       help='One or more eligible Weka clusters for training or standalone validation')
        s.add_argument('--allow-gpu-reservation', action='store_true',
                       help='Allow CPU jobs on GPU clusters while still requesting zero GPUs')
        s.add_argument('--priority', choices=('low', 'normal', 'high', 'urgent'),
                       default='urgent' if stage == 'train-model' else 'normal')
        s.add_argument('--cpus', type=int, default=RESOURCES[stage][0])
        s.add_argument('--memory', default=RESOURCES[stage][1])
        s.add_argument('--preemptible', action='store_true')
        if stage in ('train-model', 'convert-model'):
            s.add_argument('--chain-evaluation', action='store_true',
                           help='Run conversion and evaluation in the same shell/job after success')
        s.add_argument('--min-runtime', type=minimum_runtime,
                       default='30m' if stage == 'validate-model' else '3h' if stage == 'train-model' else None,
                       help='Explicit minimum allocation runtime, e.g. 3h')
        s.add_argument('--wandb-secret', help='Beaker secret name for WANDB_API_KEY; never a literal key')
        s.add_argument('--name', help='Beaker experiment name; otherwise generated uniquely')
        s.add_argument('--image', default=GPU_IMAGE if stage in (*MODEL_STAGES, 'validate-model') else CPU_IMAGE,
                       help='Beaker image ID for training; Docker image for CPU stages')
        s.add_argument('--hf-token-secret', help='Optional Beaker secret name to expose as HF_TOKEN')
        s.add_argument('--dry-run', action='store_true', help='Print JSON; no upload or job submission')
        s.add_argument('--spec-file', type=Path, help='Also save the generated JSON spec')
        s.add_argument('--receipt', type=Path, help='Save the submitted workload IDs as JSON')
    return p


def cluster_names(args):
    value = args.cluster
    return ([value] if isinstance(value, str) else value) or []


def validate_placement(args):
    if gpu_count(args) and (not cluster_names(args) or
                           (args.stage not in ('train-model', 'validate-model') and len(cluster_names(args)) != 1)):
        raise ValueError('GPU stages other than training/validation require exactly one --cluster')


def verify_gpu_clusters(beaker, args, clusters):
    """Check every candidate; training's pre-launch script adapts its batching."""
    from google.protobuf.json_format import MessageToDict
    allowed = GPU_TYPES if args.stage in ('train-model', 'validate-model') else {'GPU_TYPE_NVIDIA_H100'}
    for cluster in clusters:
        if not any(MessageToDict(node).get('nodeResources', {}).get('gpuType') in allowed
                   and len(node.node_resources.gpu_ids) >= gpu_count(args)
                   for node in beaker.node.list(cluster=cluster)):
            raise RuntimeError(f'No qualifying GPU nodes in requested cluster {cluster.name}; '
                               f'allowed GPU types: {sorted(allowed)}')


def stage_arguments(args):
    result = [args.stage, '--root', str(args.root)]
    if args.stage == 'sample-pool':
        result += ['--data-size', args.data_size, '--sampling-seed', str(args.sampling_seed)]
    elif args.stage not in ('download-data', 'prepare-validation'):
        result += ['--vocab-size', str(args.vocab_size), '--data-size', args.data_size]
    if args.stage in MODEL_STAGES:
        for name in ('validation_id', 'model_size', 'gpus', 'micro_batch_size', 'batch_size',
                     'learning_rate', 'weight_decay', 'embedding_weight_decay', 'epochs', 'seed', 'max_steps',
                     'epoch_fraction', 'branch_from', 'wandb_project', 'eval_interval'):
            value = getattr(args, name)
            if value is not None:
                result += ['--' + name.replace('_', '-'), str(value)]
        if args.no_compile:
            result.append('--no-compile')
        if args.health_check:
            result.append('--health-check')
        if args.resume_latest:
            result.append('--resume-latest')
        if args.decay_horizons:
            result += ['--decay-horizons', *map(str, args.decay_horizons)]
    if args.stage == 'validate-model':
        result += ['--model-directory', str(args.model_directory)]
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
    if getattr(args, 'chain_evaluation', False):
        env['TOKENIZER_CHAIN_EVALUATION'] = '1'
    if not gpu_job:
        env['CUDA_VISIBLE_DEVICES'] = ''
    env_vars = [{'name': k, 'value': v} for k, v in env.items()]
    if args.hf_token_secret:
        env_vars.append({'name': 'HF_TOKEN', 'secret': args.hf_token_secret})
    if args.wandb_secret:
        env_vars.append({'name': 'WANDB_API_KEY', 'secret': args.wandb_secret})
    context = {'priority': args.priority, 'preemptible': args.preemptible}
    if args.stage == 'train-model':
        # Use the current scheduling field, not the deprecated preemptible flag.
        context = {'priority': args.priority,
                   'minRuntime': args.min_runtime if args.min_runtime is not None else
                   '3h'}
    elif args.min_runtime is not None:
        context = {'priority': args.priority, 'minRuntime': args.min_runtime}
    return {
        'version': 'v2',
        'description': f'Tokenizer project: {args.stage}; source {revision}',
        'tasks': [{
            'name': 'main',
            'image': {'beaker' if args.stage in (*MODEL_STAGES, 'validate-model') else 'docker': args.image},
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
            'constraints': {'cluster': resolved_clusters or cluster_names(args) or [WEKA_CLUSTERS_PLACEHOLDER]},
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
        # The REST API expects a Go time.Duration (integer nanoseconds), unlike
        # the protobuf/read API and human-readable spec's duration string.
        min_runtime: int | None = None

    compatible = copy.deepcopy(spec)
    runtimes = [task['context'].pop('minRuntime', None) for task in compatible['tasks']]
    parsed = BeakerExperimentSpec.from_json(compatible)
    for task, runtime in zip(parsed.tasks, runtimes):
        if runtime is not None:
            task.context = RuntimeContext(**vars(task.context), min_runtime=runtime_nanoseconds(runtime))
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
            if args.stage not in RAW_STAGES or args.cluster or not rejected or rejected[1] not in clusters:
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
        validate_placement(args)
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
            eligible = discover_weka_clusters(b, workspace)
            resolved_clusters = cluster_names(args) or eligible
            if set(resolved_clusters) - set(eligible):
                raise ValueError('Requested cluster is not an accessible Weka batch cluster')
            clusters = [b.cluster.get(name) for name in resolved_clusters]
            if args.stage in (*MODEL_STAGES, 'validate-model'):
                b.image.get(args.image)
            if gpu_count(args):
                verify_gpu_clusters(b, args, clusters)
            elif (args.stage not in RAW_STAGES and not args.allow_gpu_reservation
                  and any(cluster.node_shape.gpu_count for cluster in clusters)):
                raise ValueError('CPU jobs on GPU clusters require --allow-gpu-reservation; GPU request remains 0')
            dataset = b.dataset.create(
                f'tokenizer-code-{revision[:12]}-{uuid.uuid4().hex[:8]}', archive, bootstrap,
                workspace=workspace, strip_paths=True,
                description=f'shmsw25/tokenizer-vocab-scaling source at {revision}',
            )
            spec = make_spec(args, dataset.id, revision, resolved_clusters)
            workload = create_experiment(b, args, workspace, name, spec)
            if args.receipt:
                from pipeline import write_json
                write_json(args.receipt, {'workload_id': workload.experiment.id,
                           'experiment_id': workload.experiment.id,
                           'url': b.experiment.url(workload), 'code_dataset_id': dataset.id,
                           'revision': revision, 'name': name})
            print(b.experiment.url(workload))
            print(f'Code dataset: {dataset.id} (retain for reproducibility)')
            print('Conversion and evaluation run automatically in this same job.'
                  if getattr(args, 'chain_evaluation', False) else
                  'Run the next stage only after this experiment succeeds.')


if __name__ == '__main__':
    main()
