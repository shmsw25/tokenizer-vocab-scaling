"""Prepare the pinned original EMO tokenizer on the existing DCLM 10GB pool."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from pipeline import REPO, lock, read_json, run, write_json
from dclm_data import archive_partial, sha256_file, verify_output, verify_token_stream

VOCAB = 100278
EOS = 100257
PAD = 100277
STAGE = 'prepare-dolma2-comparison'


def source_manifest():
    return read_json(REPO / 'data/dolma2_tokenizer.json')


def tokenizer_path(root):
    return Path(root) / 'tokenizers_external' / ('dolma2_' + source_manifest()['revision']) / 'tokenizer.json'


def stream_path(root):
    return Path(root) / 'tokenized_data_dclm/dolma2_10GB_100278'


def verify_tokenizer(root):
    from tokenizers import Tokenizer
    source = source_manifest()
    folder = tokenizer_path(root).parent
    for name, expected in source['files'].items():
        verify_output(folder / name, expected)
    tok = Tokenizer.from_file(str(folder / 'tokenizer.json'))
    if (tok.get_vocab_size() != VOCAB or max(tok.get_vocab().values()) + 1 != VOCAB
            or tok.token_to_id('<|endoftext|>') != EOS or tok.token_to_id('<|pad|>') != PAD):
        raise ValueError('Pinned Dolma2 vocabulary or special tokens differ from EMO')
    # Keep the native pretokenizer/normalizer; never train or modify this tokenizer.
    for text in ['Hello world!\n', 'café 中文 한국어 😀', '  x\t123.456\n', '|||EMAIL_ADDRESS|||']:
        if tok.decode(tok.encode(text, add_special_tokens=False).ids, skip_special_tokens=False) != text:
            raise ValueError('Dolma2 UTF-8 round-trip verification failed')
    return {'vocab_size': VOCAB, 'eos_token_id': EOS, 'pad_token_id': PAD,
            'padded_vocab_size': 100352, 'sha256': source['files']['tokenizer.json']['sha256'],
            'source': source, 'round_trip_verified': True}


def prepare_tokenizer(root):
    source = source_manifest()
    folder = tokenizer_path(root).parent
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in source['files'].items():
        target = folder / name
        if target.exists():
            verify_output(target, expected)
            continue
        partial = target.with_suffix(target.suffix + '.part')
        archive_partial(partial)
        url = f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}/{name}"
        for attempt in range(8):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    partial.write_bytes(response.read())
                break
            except (urllib.error.URLError, TimeoutError) as error:
                if (isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429, 500, 502, 503, 504)) or attempt == 7:
                    raise
                time.sleep(min(60, 2 ** attempt))
        verify_output(partial, expected)
        partial.replace(target)
    verified = verify_tokenizer(root)
    write_json(folder / 'pipeline.json', verified)
    print('DOLMA2_TOKENIZER_VERIFIED ' + json.dumps(verified), flush=True)
    return verified


def training_recipe(root):
    raw = Path(root) / 'raw_text_dclm/10GB/dclm_10gb.jsonl'
    pool = read_json(raw.with_name('pool.json'))
    verify_output(raw, pool['output'])
    if sha256_file(Path(pool['recipe']['heldout_file'])) != pool['recipe']['heldout_sha256']:
        raise ValueError('DCLM held-out exclusion changed')
    return {'pool': pool, 'tokenizer_source': source_manifest(),
            'tokenizer_sha256': source_manifest()['files']['tokenizer.json']['sha256'],
            'vocab_size': VOCAB, 'seed': 42, 'eval_fraction': .01,
            'eos_token_id': EOS, 'pad_token_id': PAD, 'eos_per_document': 1, 'dtype': 'uint32'}


def verify_training_stream(root, directory=None, recipe=None):
    verify_tokenizer(root)
    directory = stream_path(root) if directory is None else Path(directory)
    recipe = training_recipe(root) if recipe is None else recipe
    meta, repack = verify_token_stream(directory, recipe['pool'], VOCAB, recipe['tokenizer_sha256'], EOS)
    if meta.get('pad_token_id') != PAD or meta.get('pad_token') != '<|pad|>':
        raise ValueError('Dolma2 padding token differs from EMO')
    baseline = read_json(Path(root) / 'tokenized_data_dclm/default_10GB_32000/meta.json')
    for key in ('data_files', 'seed', 'eval_fraction'):
        if meta[key] != baseline[key]: raise ValueError(f'Dolma2 training split mismatch: {key}')
    for split in ('train', 'eval'):
        for key in ('documents', 'bytes'):
            if meta[split][key] != baseline[split][key]:
                raise ValueError(f'Dolma2 training split mismatch: {split}.{key}')
    marker = directory / 'pipeline.json'
    if marker.exists():
        info = read_json(marker)
        if info['recipe'] != recipe or info['meta'] != meta or info['repack'] != repack:
            raise ValueError('Published Dolma2 training manifest changed')
    return meta, repack


def prepare(root):
    from validation_data import tokenize_validation
    root = Path(root)
    with lock(root, 'dolma2-comparison-10gb'):
        prepare_tokenizer(root)
        recipe = training_recipe(root)
        target = stream_path(root)
        if (target / 'pipeline.json').exists():
            verify_training_stream(root, recipe=recipe)
        else:
            staging = target.with_name(target.name + '.staging')
            archive_partial(target)
            archive_partial(staging)
            print('DOLMA2_PREPARATION ' + json.dumps({'stage': 'training_tokenization',
                'total_documents': recipe['pool']['records'], 'total_text_bytes': recipe['pool']['text_bytes']}), flush=True)
            run([sys.executable, 'tokenize_data.py', '--data', recipe['pool']['output_path'],
                 '--tokenizer', tokenizer_path(root), '--output-dir', staging,
                 '--seed', '42', '--eval-fraction', '0.01', '--progress-json'])
            meta = read_json(staging / 'meta.json')
            meta.update(pad_token='<|pad|>', pad_token_id=PAD)
            write_json(staging / 'meta.json', meta)
            if training_recipe(root) != recipe: raise ValueError('Source changed during Dolma2 tokenization')
            meta, repack = verify_training_stream(root, staging, recipe)
            info = {'recipe': recipe, 'meta': meta, 'repack': repack,
                    'output_path': str(target), 'total_tokens': sum(meta[s]['tokens'] for s in ('train', 'eval'))}
            write_json(staging / 'pipeline.json', info)
            staging.replace(target)
        tokenize_validation(root, '10GB', VOCAB, dataset='dclm', tokenizer_source='dolma2')
        result = preparation_summary(root)
        print('DOLMA2_PREPARATION_VERIFIED ' + json.dumps(result), flush=True)
        return result


def preparation_summary(root):
    from validation_data import validation_metadata
    info = read_json(stream_path(root) / 'pipeline.json')
    meta, _ = verify_training_stream(root)
    return {'meta': meta, 'tokenization': info, 'tokenizer': verify_tokenizer(root),
            'validation': validation_metadata(root, '10GB', VOCAB, dataset='dclm', tokenizer_source='dolma2')}


def verify_preparation_receipt(state, node):
    """Gate GPU admission on the current worker and frozen source identities."""
    if node.get('monitor_error'):
        raise ValueError('Dolma2 CPU preparation has a monitoring error')
    summary = node['summary']
    if summary['stage'] != STAGE or summary['revision'] != node['receipt']['revision']:
        raise ValueError('Dolma2 preparation worker identity mismatch')
    cohort = state['study']['dolma2_dclm_small_10gb']
    for key, frozen in cohort['sources'].items():
        actual = state['nodes'][key]
        if (actual.get('receipt') != frozen['receipt'] or actual.get('monitor_error')
                or hashlib.sha256(json.dumps(actual['summary'], sort_keys=True).encode()).hexdigest() != frozen['summary_sha256']):
            raise ValueError('Dolma2 frozen source identity mismatch')
    tokenizer = summary['tokenizer']
    if (tokenizer['source'] != cohort['tokenizer_source'] or tokenizer['vocab_size'] != VOCAB
            or tokenizer['eos_token_id'] != EOS or tokenizer['pad_token_id'] != PAD
            or tokenizer['sha256'] != source_manifest()['files']['tokenizer.json']['sha256']):
        raise ValueError('Dolma2 prepared tokenizer identity mismatch')
    meta = summary['meta']
    baseline = state['nodes']['dclm-tokens-10gb-32000']['summary']['meta']
    if meta['vocab_size'] != VOCAB or meta['eos_token_id'] != EOS or meta['pad_token_id'] != PAD:
        raise ValueError('Dolma2 prepared training token IDs mismatch')
    for key in ('data_files', 'seed', 'eval_fraction'):
        if meta[key] != baseline[key]: raise ValueError('Dolma2 prepared training split mismatch')
    for split in ('train', 'eval'):
        if any(meta[split][k] != baseline[split][k] for k in ('bytes', 'documents')):
            raise ValueError('Dolma2 prepared training documents mismatch')
    validation = summary['validation']
    corpus = state['nodes'][state['study']['validation_node']]['summary']['validation']
    recipe = validation['recipe']
    if (recipe['documents_sha256'] != corpus['sha256'] or validation['bytes'] != corpus['text_bytes']
            or validation['documents'] != corpus['documents'] or recipe['tokenizer_source'] != source_manifest()
            or recipe['tokenizer_sha256'] != tokenizer['sha256'] or recipe['vocab_size'] != VOCAB
            or recipe['validation_id'] != 'dolma100m-v1' or recipe['pool'] != '10GB'
            or recipe['tokenizer_dataset'] != 'dclm'):
        raise ValueError('Dolma2 prepared validation identity mismatch')
    return True


def authorize_two_runs(state):
    """Exactly two new roots and one shared CPU prerequisite; retain all old nodes."""
    from campaign import now, prepare_chains
    from runtime_batch import GPU_CLUSTERS
    study, nodes = state['study'], state['nodes']
    cohort = 'dolma2_dclm_small_10gb'
    if cohort in study: raise ValueError('Dolma2 comparisons already authorized')
    source_keys = ['dclm-tokens-10gb-32000', study['validation_node']]
    sources = {}
    for key in source_keys:
        n = nodes[key]
        if (n['status'] != 'succeeded' or n.get('monitor_error')
                or n['summary']['revision'] != n['receipt']['revision'] or n['summary']['stage'] != n['stage']):
            raise ValueError(f'Unverified Dolma2 prerequisite: {key}')
        sources[key] = {'receipt': copy.deepcopy(n['receipt']),
                        'summary_sha256': hashlib.sha256(json.dumps(n['summary'], sort_keys=True).encode()).hexdigest()}
    prep_key = 'prepare-dolma2-dclm-10gb'
    common = ['--root', study['root'], '--dataset', 'dclm', '--tokenizer-source', 'dolma2',
              '--data-size', '10GB', '--vocab-size', str(VOCAB)]
    additions = {prep_key: dict(stage=STAGE, dataset='dclm', tokenizer_source='dolma2',
        arguments=[STAGE, *common], dependencies=source_keys, status='pending',
        data_size='10GB', vocab_size=VOCAB, model_size=None, expected_gpus=0,
        min_runtime='1h', priority='normal', placement_clusters=['ai2/saturn', 'ai2/neptune'],
        allow_gpu_reservation=True, study_role='explicit_dolma2_comparison')}
    model_args = [*common, '--model-size', 'small', '--gpus', '8', '--batch-size', '128',
        '--micro-batch-size', '16', '--learning-rate', '0.0005', '--weight-decay', '0.1',
        '--embedding-weight-decay', '0.1', '--epochs', '1', '--epoch-fraction', '1', '--seed', '42',
        '--validation-id', 'dolma100m-v1', '--wandb-project', 'tokenizer-scaling',
        '--eval-interval', '1000', '--training-recipe', 'emo-dense-v1']
    roots, required = [], []
    for filtered in (False, True):
        suffix = 'dclm-small-10gb-dolma2-lr0p0005-emo-dense-v1' + ('-repfilter' if filtered else '')
        arguments = [*model_args, *(['--repetition-filter', 'emo-v1'] if filtered else [])]
        deps = [prep_key, *source_keys]
        for prefix, stage in [('train', 'train-model'), ('convert', 'convert-model'), ('eval', 'evaluate-model')]:
            key = prefix + '-' + suffix
            additions[key] = dict(stage=stage, dataset='dclm', tokenizer_source='dolma2',
                arguments=[stage, *arguments], dependencies=deps, status='pending',
                data_size='10GB', vocab_size=VOCAB, model_size='small', epoch_fraction=1.,
                expected_gpus=8 if prefix == 'train' else int(prefix == 'eval'),
                min_runtime='1h', priority='urgent', study_role='explicit_dolma2_comparison',
                placement_clusters=list(GPU_CLUSTERS) if prefix == 'train' else ['ai2/jupiter'],
                validation_stream_node=prep_key)
            if prefix == 'train':
                additions[key].update(chain_evaluation=True, wandb_secret='SEWONM_WANDB_API_KEY')
                roots.append(key)
            if prefix == 'eval': required.append(key)
            deps = [key]
    if set(additions) & set(nodes): raise ValueError('Dolma2 comparison nodes already exist')
    authorized_at = now()
    nodes.update(additions)
    for key in roots:
        study.setdefault('training_submission_exceptions', {})[key] = dict(
            source='explicit_user_request', authorized_at=authorized_at,
            arguments_sha256=hashlib.sha256(json.dumps(nodes[key]['arguments'], separators=(',', ':')).encode()).hexdigest(),
            reason='Two fresh Dolma2-tokenizer DCLM 10GB small EMO runs, one unfiltered and one RepFilter')
    study[cohort] = dict(authorized_at=authorized_at, runs=roots, required=required,
        preparation_node=prep_key, sources=sources, tokenizer_source=source_manifest(),
        scope='Exactly two fresh runs with chained evaluations; no retries or replacement training without authorization')
    study['required'] = sorted(set(study['required']) | set(required))
    prepare_chains(state)
    return study[cohort]
