"""Run one pipeline stage inside a Beaker job; also usable for small local checks."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from decimal import Decimal

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = '/oe-training-default/sewonm/tokenizer_project'
REFERENCE_VOCAB = 32000
# Infinite-compute recipe: 24 steps at a global batch of 1024 sequences.
WARMUP_SEQUENCES = 24 * 1024
DEFAULT_BATCH_SIZE = 128
MODEL_MICROBATCHES = {'small': 16, 'medium': 16, 'medium_large': 8}
MODEL_STAGES = ('train-model', 'convert-model', 'evaluate-model')
RAW_STAGES = ('download-data', 'sample-pool')
POOL_SIZES = ('2.5GB', '5GB', '10GB', '20GB', '40GB', '80GB')
TRAINING_POOLS = ('5GB', '10GB', '20GB')
STAGES = (*RAW_STAGES, 'train-tokenizer', 'tokenize-data', *MODEL_STAGES)
EVALS = ('blimp', 'arc_easy', 'hellaswag', 'piqa', 'dolma_bpb')


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return value


def data_size(value):
    if not re.fullmatch(r'\d+(?:\.\d+)?GB', value):
        raise argparse.ArgumentTypeError('use a positive size in GB, e.g. 2.5GB')
    number = Decimal(value[:-2])
    if number <= 0:
        raise argparse.ArgumentTypeError('data size must be positive')
    return f'{number.normalize():f}GB'


def add_stage_arguments(parser, stage):
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    if stage == 'sample-pool':
        parser.add_argument('--data-size', required=True, type=data_size)
        parser.add_argument('--sampling-seed', type=int, default=42,
                            help='Base seed; a distinct deterministic seed is derived for each size')
    elif stage != 'download-data':
        parser.add_argument('--vocab-size', required=True, type=positive_int)
        parser.add_argument('--data-size', type=data_size, default='100GB')
    if stage in MODEL_STAGES:
        parser.add_argument('--model-size', default='small', help='Name of configs/<model-size>.json')
        parser.add_argument('--gpus', type=positive_int, default=8)
        parser.add_argument('--micro-batch-size', type=positive_int,
                            help='Per-GPU sequences; defaults to 16/16/8 by model size')
        parser.add_argument('--batch-size', type=positive_int, default=DEFAULT_BATCH_SIZE,
                            help='Global sequences per optimizer step, independent of GPU count')
        parser.add_argument('--learning-rate', type=positive_float)
        parser.add_argument('--weight-decay', type=float)
        parser.add_argument('--epochs', type=positive_int, default=1)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--max-steps', type=positive_int,
                            help='Optional short run; overrides the epoch budget')
        parser.add_argument('--no-compile', action='store_true')
        parser.add_argument('--epoch-fraction', type=positive_float, default=1.0)
        parser.add_argument('--decay-horizons', type=float, nargs='+', default=[])
        parser.add_argument('--branch-from', type=Path)
        parser.add_argument('--wandb-project')
        parser.add_argument('--health-check', action='store_true')
        parser.add_argument('--resume-latest', action='store_true',
                            help='Require and restore the latest full-state checkpoint; never start over')
        parser.add_argument('--eval-interval', type=positive_int)
    if stage == 'evaluate-model':
        parser.add_argument('--evals', nargs='+', choices=EVALS, default=list(EVALS))


def validate_args(args):
    if not args.root.is_absolute():
        raise ValueError('--root must be absolute')
    if args.stage == 'download-data':
        return
    if args.stage == 'sample-pool':
        if Decimal(args.data_size[:-2]) > 100:
            raise ValueError('Subsample size must not exceed the 100GB parent')
        return
    if args.vocab_size < 257:
        raise ValueError('--vocab-size must include 256 byte symbols and EOS (at least 257)')
    if args.stage in MODEL_STAGES:
        if not 0 < args.epoch_fraction <= 1 or any(not 0 < f < 1 for f in args.decay_horizons):
            raise ValueError('Epoch fraction must be in (0, 1]; decay horizons in (0, 1)')
        if args.branch_from and (args.epoch_fraction == 1 or args.decay_horizons):
            raise ValueError('Decay branches require a fractional horizon and no child horizons')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.model_size):
            raise ValueError('--model-size must be a config name, without directories')
        if not (REPO / 'configs' / f'{args.model_size}.json').is_file():
            raise ValueError(f'Missing configs/{args.model_size}.json')
        if args.micro_batch_size is None:
            args.micro_batch_size = MODEL_MICROBATCHES[args.model_size]
        if WARMUP_SEQUENCES % args.batch_size:
            raise ValueError('--batch-size must divide 24576 to preserve the fixed warmup')
        if args.batch_size % (args.gpus * args.micro_batch_size):
            raise ValueError('--batch-size must be divisible by --gpus * --micro-batch-size')
        if args.weight_decay is not None and (not math.isfinite(args.weight_decay) or args.weight_decay < 0):
            raise ValueError('--weight-decay must be finite and nonnegative')


def dataset_name(args, vocab=None):
    return f'default_{args.data_size}_{args.vocab_size if vocab is None else vocab}'


def tokenized_dir(args, vocab=None):
    return args.root / 'tokenized_data' / dataset_name(args, vocab)


def raw_file(args):
    if args.data_size == '100GB':
        return args.root / 'raw_text/100GB/training/dolma3_100gb.jsonl'
    return args.root / 'raw_text' / args.data_size / f'dolma3_{args.data_size.lower()}.jsonl'


def tokenizer_file(args):
    return args.root / 'tokenizers' / dataset_name(args) / 'tokenizer.json'


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    partial.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_signature(path):
    st = path.stat()
    if not st.st_size:
        raise ValueError(f'Empty raw text file: {path}')
    return {'path': str(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


@contextmanager
def lock(root, name):
    locks = root / '.locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f'{name}.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another job is already writing {name}') from None
        yield


def run(command):
    print('Running:', ' '.join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True, cwd=REPO)


def download_data(args):
    full = args.root / 'raw_text/250GB/dolma3_250gb.jsonl'
    sample = args.root / 'raw_text/100GB/dolma3_100gb.jsonl'
    with lock(args.root, 'raw-text'):
        for path in (full, sample):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not full.exists():
            run([sys.executable, 'data/download_dolma.py', '--output', full])
        raw_signature(full)
        if not sample.exists():
            run([sys.executable, 'data/repeat_dolma.py', '--input', full,
                 '--sample-fraction', '0.4', '--seed', '42', '--output', sample])
        raw_signature(sample)
        write_json(args.root / 'raw_text/provenance.json', {
            'upstream': read_json(REPO / 'data/dolma_text_estimate_metadata.json'),
            'target_text_bytes': 250_000_000_000, 'sample_fraction': 0.4,
            'sampling_seed': 42, 'repetitions': 1,
            'full': raw_signature(full), 'sample': raw_signature(sample),
        })
        print(f'Raw data ready: {sample}', flush=True)


def train_tokenizer(args):
    target = tokenizer_file(args)
    signature = raw_signature(raw_file(args))
    pool = pool_provenance(args)
    marker = target.with_name('pipeline.json')
    with lock(args.root, f'tokenizer-{dataset_name(args)}'):
        if target.exists():
            info = read_json(marker)
            if info['raw'] != signature or info['sha256'] != digest(target):
                raise ValueError('Existing tokenizer does not match its source/provenance')
            print(f'Tokenizer already complete: {target}', flush=True)
            return
        # Publish the tokenizer only after its provenance is complete. An
        # interrupted attempt can replace this private staging file on retry.
        staging = target.with_name('tokenizer.staging.json')
        run([sys.executable, 'train_tokenizer.py', '--data', raw_file(args),
             '--vocab-size', args.vocab_size, '--output', staging, '--overwrite'])
        if raw_signature(raw_file(args)) != signature:
            raise ValueError('Raw text changed during tokenizer training')
        write_json(marker, {'raw': signature, 'vocab_size': args.vocab_size,
                            'sha256': digest(staging), 'pool': pool})
        staging.replace(target)


def completed_meta(directory, expected_vocab):
    meta = read_json(directory / 'meta.json')
    if meta['vocab_size'] != expected_vocab:
        raise ValueError(f'Wrong vocabulary in {directory}: {meta["vocab_size"]}')
    ratio = meta['tokens_per_byte']
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f'Invalid tokens_per_byte in {directory}')
    for split in ('train', 'eval'):
        if (directory / f'{split}.bin').stat().st_size != meta[split]['tokens'] * 4:
            raise ValueError(f'{split}.bin size does not match meta.json in {directory}')
    if not (directory / 'tokenizer.json').is_file():
        raise FileNotFoundError(directory / 'tokenizer.json')
    return meta


def tokenize_data(args):
    target = tokenized_dir(args)
    expected = {'raw': raw_signature(raw_file(args)), 'tokenizer_sha256': digest(tokenizer_file(args))}
    pool_provenance(args)
    marker = target / 'pipeline.json'
    with lock(args.root, f'tokenized-{dataset_name(args)}'):
        if (target / 'meta.json').exists():
            completed_meta(target, args.vocab_size)
            if read_json(marker) != expected:
                raise ValueError('Existing tokenized data has a different source/tokenizer')
            print(f'Tokenized data already complete: {target}', flush=True)
            return
        # An incomplete attempt is rewritten from the start while holding the lock.
        # Publish provenance first; meta.json remains the completion marker.
        write_json(marker, expected)
        run([sys.executable, 'tokenize_data.py', '--data', raw_file(args),
             '--tokenizer', tokenizer_file(args), '--output-dir', target,
             '--seed', '42', '--eval-fraction', '0.01'])
        if raw_signature(raw_file(args)) != expected['raw']:
            (target / 'meta.json').unlink()
            raise ValueError('Raw text changed during tokenization')
        completed_meta(target, args.vocab_size)


def training_plan(args):
    validate_args(args)
    data = tokenized_dir(args)
    reference = tokenized_dir(args, REFERENCE_VOCAB)
    meta = completed_meta(data, args.vocab_size)
    baseline = completed_meta(reference, REFERENCE_VOCAB)
    # The measured ratios must come from precisely the same corpus/split.
    for key in ('data_files', 'seed', 'eval_fraction'):
        if meta[key] != baseline[key]:
            raise ValueError(f'Target and 32k reference differ in {key}')
    for split in ('train', 'eval'):
        for key in ('documents', 'bytes'):
            if meta[split][key] != baseline[split][key]:
                raise ValueError(f'Target and 32k reference differ in {split}.{key}')
    for directory in (data, reference):
        provenance = read_json(directory / 'pipeline.json')
        if provenance['raw'] != raw_signature(raw_file(args)):
            raise ValueError(f'Raw text no longer matches {directory}')
        if provenance['tokenizer_sha256'] != digest(directory / 'tokenizer.json'):
            raise ValueError(f'Tokenizer has changed in {directory}')
    seq = round(4096 * meta['tokens_per_byte'] / baseline['tokens_per_byte'])
    if seq < 2:
        raise ValueError('Scaled sequence length is less than 2')
    # The loader drops an incomplete final global batch; fail before using a GPU
    # for a run too small to contain one complete optimizer step.
    if meta['train']['tokens'] // seq < args.batch_size or meta['eval']['tokens'] < seq:
        raise ValueError('Not enough train/eval tokens for the scaled sequence length and batch')
    cfg = read_json(REPO / 'configs' / f'{args.model_size}.json')
    cfg['seq_length'] = seq
    t = cfg['training']
    t.update(epochs=args.epochs, seed=args.seed, micro_batch_size=args.micro_batch_size,
             gradient_accumulation_steps=args.batch_size // (args.gpus * args.micro_batch_size))
    t.pop('max_steps', None)
    if args.max_steps is not None:
        t['max_steps'] = args.max_steps
    if args.learning_rate is not None:
        t['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        t['weight_decay'] = args.weight_decay
    if args.no_compile:
        t['compile'] = False
    # Fixed data warmup from infinite-compute; context matching also matches bytes.
    t.pop('warmup_fraction', None)
    t['warmup_steps'] = WARMUP_SEQUENCES // args.batch_size
    t['decay_fraction'] = 0.1
    t['decay_min_lr'] = 0.0
    epoch_steps = meta['train']['tokens'] // (seq * args.batch_size)
    if args.epoch_fraction != 1:
        t['max_steps'] = math.floor(epoch_steps * args.epoch_fraction)
        if t['max_steps'] - round(t['max_steps'] * .1) <= t['warmup_steps']:
            raise ValueError('Fractional horizon is too short for fixed warmup and decay')
    if args.decay_horizons:
        t['decay_horizons'] = sorted(set(args.decay_horizons))
    if args.wandb_project:
        t.update(report_to='wandb', wandb_project=args.wandb_project)
    if args.health_check:
        t['health_check'] = True
    if args.eval_interval:
        t['eval_steps'] = args.eval_interval
    if args.branch_from:
        t['branch_from'] = str(args.branch_from)
    cfg['pipeline'] = {
        'scheduler': 'WSD', 'reference_vocab_size': REFERENCE_VOCAB,
        'reference_sequence_length': 4096, 'reference_tokens_per_byte': baseline['tokens_per_byte'],
        'tokens_per_byte': meta['tokens_per_byte'], 'global_batch_size': args.batch_size,
        'gpus': args.gpus, 'tokenizer_sha256': digest(data / 'tokenizer.json'),
        'raw': raw_signature(raw_file(args)),
    }
    if pool := pool_provenance(args):
        cfg['pipeline']['pool'] = pool
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    # AdamW uses one default parameter group: embeddings receive the same WD.
    tags = ('_wt' if cfg['model'].get('tie_word_embeddings', False) else '')
    tags += '_EmbedWD' if t['weight_decay'] > 0 else ''
    run_name = (f'wsd_lr{t["learning_rate"]:g}_wd{t["weight_decay"]:g}_bs{args.batch_size}_'
                f'seq{seq}_seed{args.seed}{tags}_{fingerprint}')
    output = args.root / 'models' / args.model_size / dataset_name(args) / run_name
    return cfg, output


def train_model(args):
    cfg, output = training_plan(args)
    with lock(args.root, f'model-{args.model_size}-{dataset_name(args)}-{output.name}'):
        if (output / 'training_summary.json').is_file():
            model_artifacts(args)  # Check config and final checkpoint before reusing a completed run.
            print(f'Training already complete: {output}', flush=True)
            return
        config_path = output / 'launch_config.json'
        if config_path.exists() and read_json(config_path) != cfg:
            raise ValueError('Refusing to resume with a changed config')
        write_json(config_path, cfg)
        if args.branch_from:
            parent = read_json(args.branch_from / 'launch_config.json')
            # Only the horizon, checkpoint source, and diagnostic cadence may differ.
            def scientific_config(value):
                value = json.loads(json.dumps(value))
                for key in ('max_steps', 'decay_horizons', 'branch_from', 'eval_steps'):
                    value['training'].pop(key, None)
                return value
            if scientific_config(parent) != scientific_config(cfg):
                raise ValueError('Decay branch differs from its trunk scientific configuration')
            record = read_json(args.branch_from / 'decay_branches.json')[f'{args.epoch_fraction:g}']
            total = cfg['training']['max_steps']
            boundary = total - round(total * .1)
            if record['step'] != boundary or record['total_steps'] != total:
                raise ValueError('Decay branch boundary does not match its requested horizon')
        # The exact source revision is also available in the Beaker experiment
        # and its immutable code dataset; retain it next to the checkpoints.
        with (output / 'source_revisions.txt').open('a') as history:
            history.write(os.environ.get('TOKENIZER_CODE_REVISION', 'local') + '\n')
        print(f'Model output: {output}', flush=True)
        print(f'Scaled sequence length: {cfg["seq_length"]}', flush=True)
        resume_args = []
        if args.resume_latest:
            checkpoint = latest_checkpoint(output)
            # Validate all ranks before the distributed restore, which also validates tensor metadata.
            required = [checkpoint / 'train' / f'rank{rank}.pt' for rank in range(args.gpus)]
            required += list((checkpoint / 'model_and_optim').glob('*.distcp'))
            if len(required) == args.gpus or any(not p.is_file() or not p.stat().st_size for p in required):
                raise ValueError(f'Incomplete full-state resume checkpoint: {checkpoint}')
            record = {'checkpoint': str(checkpoint), 'step': int(checkpoint.name[4:]),
                      'learning_rate': cfg['training']['learning_rate'],
                      'source_revision': os.environ.get('TOKENIZER_CODE_REVISION', 'local')}
            write_json(output / 'resume_request.json', record)
            old_stop = output / 'health_stop.json'
            if old_stop.is_file():
                archive = output / 'health_stop_history'
                archive.mkdir(exist_ok=True)
                old_stop.rename(archive / f'{time.time_ns()}.json')
            print('RESUME_REQUEST ' + json.dumps(record), flush=True)
            resume_args = ['--resume-from', str(checkpoint)]
        started = time.monotonic()
        run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
             f'--nproc_per_node={args.gpus}', 'train.py', '--config', config_path,
             '--data-dir', tokenized_dir(args), '--output-dir', output,
             '--run-name', f'{args.model_size}_{dataset_name(args)}_{output.name}', *resume_args])
        checkpoint = latest_checkpoint(output)
        write_json(output / 'training_summary.json', {
            'model_size': args.model_size, 'vocab_size': args.vocab_size, 'data_size': args.data_size,
            'model_dir': str(output), 'checkpoint': str(checkpoint), 'config': cfg,
            'elapsed_seconds_this_attempt': time.monotonic() - started,
            'training': read_json(output / 'training_metrics.json'),
        })


def latest_checkpoint(output):
    checkpoints = [p for p in output.glob('step*') if p.name[4:].isdigit()
                   and (p / '.metadata.json').is_file()
                   and (p / 'config.json').is_file()
                   and (p / 'model_and_optim/.metadata').is_file()]
    if not checkpoints:
        raise ValueError(f'No complete checkpoint in {output}')
    return max(checkpoints, key=lambda p: int(p.name[4:]))


def model_artifacts(args):
    cfg, output = training_plan(args)
    summary = read_json(output / 'training_summary.json')
    if summary['config'] != cfg:
        raise ValueError('Training summary does not match requested configuration')
    checkpoint = latest_checkpoint(output)
    if str(checkpoint) != summary['checkpoint']:
        raise ValueError('Latest checkpoint differs from completed training')
    export_root = output / 'hf'
    exported = export_root / checkpoint.parent.name / checkpoint.name
    return output, checkpoint, export_root, exported


def convert_model(args):
    output, checkpoint, export_root, exported = model_artifacts(args)
    with lock(args.root, f'conversion-{args.model_size}-{dataset_name(args)}-{output.name}'):
        if not exported.exists():
            run([sys.executable, 'eval/checkpoint_converter.py', checkpoint,
                 '--output-dir', export_root])
        conversion = read_json(exported / 'conversion.json')
        if conversion['source_checkpoint'] != str(checkpoint):
            raise ValueError('Export came from a different checkpoint')
        # Scan the raw corpus once on CPUs; every evaluation reuses this sample.
        sys.path.insert(0, str(REPO / 'eval'))
        import eval_suite
        eval_suite.DOLMA_DATA_FILE = raw_file(args)
        eval_suite.DOLMA_SAMPLE_DIR = args.root / '.cache/dolma_samples'
        pool = pool_provenance(args)
        eval_suite.DOLMA_HELDOUT_FILE = Path(pool['heldout_file']) if pool else None
        sample = eval_suite.dolma_sample()
        write_json(output / 'conversion_summary.json', {
            'exported': str(exported), 'conversion': conversion,
            'heldout_documents': len(sample),
            'heldout_file': pool['heldout_file'] if pool else None,
        })


def evaluate_model(args):
    output, checkpoint, _, exported = model_artifacts(args)
    read_json(output / 'conversion_summary.json')
    with lock(args.root, f'evaluation-{args.model_size}-{dataset_name(args)}-{output.name}'):
        directory = output / 'evaluation'
        directory.mkdir(parents=True, exist_ok=True)
        metrics = {}
        pool = pool_provenance(args)
        heldout_args = ['--dolma-heldout-file', pool['heldout_file']] if pool else []
        for name in args.evals:
            result = directory / f'{name}.json'
            if not result.exists():
                run([sys.executable, 'eval/eval_suite.py', exported, name,
                     '--dolma-sample-dir', args.root / '.cache/dolma_samples',
                     '--output-dir', directory, *heldout_args])
            value = read_json(result)
            if value['eval'] != name or value['model_checkpoint'] != str(exported):
                raise ValueError(f'Evaluation provenance mismatch: {result}')
            metrics[name] = value
        write_json(output / 'evaluation_summary.json', {
            'model_size': args.model_size, 'vocab_size': args.vocab_size, 'data_size': args.data_size,
            'checkpoint': str(checkpoint), 'exported': str(exported), 'metrics': metrics,
            'training': read_json(output / 'training_summary.json'),
        })


def publish_summary(args):
    """Expose small manifests through Beaker results; large artifacts stay on Weka."""
    summary = {'stage': args.stage, 'revision': os.environ.get('TOKENIZER_CODE_REVISION', 'local')}
    if args.stage == 'download-data':
        summary['data'] = read_json(args.root / 'raw_text/provenance.json')
    elif args.stage == 'sample-pool':
        summary['pool'] = pool_provenance(args)
    elif args.stage == 'train-tokenizer':
        summary['tokenizer'] = read_json(tokenizer_file(args).with_name('pipeline.json'))
    elif args.stage == 'tokenize-data':
        summary['meta'] = completed_meta(tokenized_dir(args), args.vocab_size)
    else:
        _, output = training_plan(args)
        filename = {'train-model': 'training', 'convert-model': 'conversion',
                    'evaluate-model': 'evaluation'}[args.stage]
        summary.update(read_json(output / f'{filename}_summary.json'))
    if Path('/results').is_dir():
        write_json(Path('/results') / f'{args.stage}.json', summary)
        write_json(Path('/results/summary.json'), summary)
    return summary


def pool_provenance(args):
    """Require a completed, immutable sample manifest for the new smaller pools."""
    if Decimal(args.data_size[:-2]) > 100:
        return None
    info = read_json(raw_file(args).with_name('pool.json'))
    if info['output'] != raw_signature(raw_file(args)):
        raise ValueError('Pool contents no longer match their manifest')
    if digest(Path(info['heldout_file'])) != info['heldout_sha256']:
        raise ValueError('Shared held-out documents changed')
    return info


def sample_pool(args):
    from sample_pool import sample_pool as sample
    sample(args.root, args.data_size, args.sampling_seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    for stage in STAGES:
        add_stage_arguments(sub.add_parser(stage), stage)
    args = parser.parse_args()
    validate_args(args)
    args.root.mkdir(parents=True, exist_ok=True)
    {'download-data': download_data, 'sample-pool': sample_pool, 'train-tokenizer': train_tokenizer,
     'tokenize-data': tokenize_data, 'train-model': train_model,
     'convert-model': convert_model, 'evaluate-model': evaluate_model}[args.stage](args)
    publish_summary(args)


if __name__ == '__main__':
    main()
