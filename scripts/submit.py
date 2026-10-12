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

from pipeline import REPO, STAGES, add_stage_arguments, validate_args

# Existing ai2/flex2 image, verified by Beaker image lookup on 2026-10-12.
GPU_IMAGE = '01KAXVQ0YN242MQ9YEQDQQK512'  # tylerr/olmo-core-tch280cu128-2025-11-25
CPU_IMAGE = 'python:3.12-slim-bookworm'
RESOURCES = {
    'download-data': (4, '32 GiB'),
    'train-tokenizer': (64, '128 GiB'),
    'tokenize-data': (16, '64 GiB'),
    'train-model': (8, '64 GiB'),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    for stage in STAGES:
        s = sub.add_parser(stage)
        add_stage_arguments(s, stage)
        s.add_argument('--workspace', default=os.environ.get('BEAKER_WORKSPACE', 'ai2/flex2'))
        s.add_argument('--cluster', default=os.environ.get('BEAKER_CLUSTER', 'ai2/jupiter'))
        s.add_argument('--priority', choices=('low', 'normal', 'high', 'urgent'), default='normal')
        s.add_argument('--cpus', type=int, default=RESOURCES[stage][0])
        s.add_argument('--memory', default=RESOURCES[stage][1])
        s.add_argument('--preemptible', action='store_true')
        s.add_argument('--name', help='Beaker experiment name; otherwise generated uniquely')
        s.add_argument('--image', default=GPU_IMAGE if stage == 'train-model' else CPU_IMAGE,
                       help='Beaker image ID for training; Docker image for CPU stages')
        s.add_argument('--hf-token-secret', help='Optional Beaker secret name to expose as HF_TOKEN')
        s.add_argument('--dry-run', action='store_true', help='Print JSON; no upload or job submission')
        s.add_argument('--spec-file', type=Path, help='Also save the generated JSON spec')
    return p


def stage_arguments(args):
    result = [args.stage, '--root', str(args.root)]
    if args.stage != 'download-data':
        result += ['--vocab-size', str(args.vocab_size), '--data-size', args.data_size]
    if args.stage == 'train-model':
        for name in ('model_size', 'gpus', 'micro_batch_size', 'batch_size',
                     'learning_rate', 'weight_decay', 'epochs', 'seed', 'max_steps'):
            value = getattr(args, name)
            if value is not None:
                result += ['--' + name.replace('_', '-'), str(value)]
        if args.no_compile:
            result.append('--no-compile')
    return result


def make_spec(args, code_dataset='COMMITTED_CODE_DATASET', revision='HEAD'):
    gpu_job = args.stage == 'train-model'
    env = {
        'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'true',
        'RAYON_NUM_THREADS': str(args.cpus),
        'OMP_NUM_THREADS': str(max(1, args.cpus // (args.gpus if gpu_job else 1))),
        'HF_HOME': str(args.root / '.cache/huggingface'),
        'TOKENIZER_CODE_REVISION': revision,
    }
    if not gpu_job:
        env['CUDA_VISIBLE_DEVICES'] = ''
    env_vars = [{'name': k, 'value': v} for k, v in env.items()]
    if args.hf_token_secret:
        env_vars.append({'name': 'HF_TOKEN', 'secret': args.hf_token_secret})
    return {
        'version': 'v2',
        'description': f'Tokenizer project: {args.stage}; source {revision}',
        'tasks': [{
            'name': 'main',
            'image': {'beaker' if gpu_job else 'docker': args.image},
            'command': ['bash', '/code/bootstrap.sh'],
            'arguments': stage_arguments(args),
            'envVars': env_vars,
            'datasets': [
                {'mountPath': '/code', 'source': {'beaker': code_dataset}},
                {'mountPath': '/oe-training-default', 'source': {'weka': 'oe-training-default'}},
            ],
            'result': {'path': '/results'},
            'resources': {'cpuCount': args.cpus, 'memory': args.memory,
                          'gpuCount': args.gpus if gpu_job else 0,
                          'sharedMemory': '8 GiB' if gpu_job else '1 GiB'},
            'constraints': {'cluster': [args.cluster]},
            'context': {'priority': args.priority, 'preemptible': args.preemptible},
        }],
    }


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def save_spec(spec, destination):
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(spec, indent=2) + '\n')


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
    from beaker import Beaker, BeakerExperimentSpec

    if git('status', '--porcelain', '--untracked-files=normal'):
        p.error('Commit all project changes before submission; jobs use git archive HEAD')
    # Parse before any upload/launch. The SDK consumes its input dictionary.
    BeakerExperimentSpec.from_json(copy.deepcopy(spec))
    name = args.name or f'tokenizer-{args.stage}-{uuid.uuid4().hex[:12]}'
    with Beaker.from_env(default_workspace=args.workspace, check_for_upgrades=False) as b:
        # Read-only checks precede the source upload.
        workspace = b.workspace.get(args.workspace)
        b.cluster.get(args.cluster)
        if args.stage == 'train-model':
            b.image.get(args.image)
        with tempfile.TemporaryDirectory(prefix='tokenizer-source-') as temporary:
            folder = Path(temporary)
            archive = folder / 'repo.tar.gz'
            subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar.gz',
                            f'--output={archive}', revision], check=True)
            bootstrap = folder / 'bootstrap.sh'
            bootstrap.write_text(git('show', f'{revision}:scripts/bootstrap.sh') + '\n')
            dataset = b.dataset.create(
                f'tokenizer-code-{revision[:12]}-{uuid.uuid4().hex[:8]}', archive, bootstrap,
                workspace=workspace, strip_paths=True,
                description=f'shmsw25/tokenizer-vocab-scaling source at {revision}',
            )
            spec = make_spec(args, dataset.id, revision)
            save_spec(spec, args.spec_file)
            workload = b.experiment.create(name=name, workspace=workspace,
                                          spec=BeakerExperimentSpec.from_json(copy.deepcopy(spec)))
        print(b.experiment.url(workload))
        print(f'Code dataset: {dataset.id} (retain for reproducibility)')
        print('Run the next stage only after this experiment succeeds.')


if __name__ == '__main__':
    main()
