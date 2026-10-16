"""Pinned, document-uniform DCLM sampling and CPU-only tokenizer preparation.

No model training is dispatched here. Sampled documents, tokenizers and token
streams persist on Weka; compressed input shards are transient local files.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import random
import struct
import tempfile
import time
import urllib.request
import uuid

import numpy as np
import zstandard

from pipeline import REPO, lock, raw_signature, read_json, run, write_json
from sample_pool import HELDOUT_SEED, pool_seed

BLOCK = 8 * 1024 * 1024
POOLS = {'5GB': 5_000_000_000, '10GB': 10_000_000_000, '20GB': 20_000_000_000}
SOURCE = REPO / 'data/dclm_source.json'


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(BLOCK):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def emit(**values):
    print('DCLM_PROGRESS ' + json.dumps({'time': time.time(), **values}, sort_keys=True), flush=True)


def archive_partial(path):
    """Keep interrupted output as evidence before a whole-unit retry."""
    path = Path(path)
    if path.exists():
        saved = path.parent / 'partial_history' / f'{path.name}.{uuid.uuid4().hex}'
        saved.parent.mkdir(parents=True, exist_ok=True)
        path.replace(saved)


def verify_output(path, info):
    if Path(path).stat().st_size != info['bytes'] or sha256_file(path) != info['sha256']:
        raise ValueError(f'Published DCLM output changed: {path}')


def file_info(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': sha256_file(path)}


def load_inventory(folder, source):
    marker = folder / 'source_inventory.json'
    if marker.exists():
        info = read_json(marker)
        if info['source'] != source:
            raise ValueError('DCLM source or sampling policy changed')
        files = info['files']
    else:
        from huggingface_hub import HfApi
        files = []
        for item in HfApi().list_repo_tree(source['repo'], repo_type='dataset',
                                         revision=source['revision'], recursive=True):
            if item.path.endswith('.jsonl.zst'):
                if not item.lfs or not item.lfs.sha256:
                    raise ValueError(f'Missing upstream checksum: {item.path}')
                files.append({'path': item.path, 'size': item.size, 'blob_id': item.blob_id,
                              'sha256': item.lfs.sha256})
        files.sort(key=lambda x: x['path'])
    identity = [{k: f[k] for k in ('path', 'size', 'blob_id')} for f in files]
    if (len(files) != source['data_files'] or sum(f['size'] for f in files) != source['compressed_bytes']
            or json_hash(identity) != source['inventory_sha256']):
        raise ValueError('Pinned DCLM inventory count, size or hash does not match')
    if not marker.exists():
        write_json(marker, {'source': source, 'files': files})
    return files


def sample_shard(folder, source, item, url=None):
    """Bernoulli(p) per document, independent deterministic RNG for each shard."""
    folder = Path(folder)
    identity = hashlib.sha256(item['path'].encode()).hexdigest()
    target = folder / f'{identity}.jsonl'
    index = folder / f'{identity}.index.bin'
    marker = folder / f'{identity}.json'
    recipe = {'algorithm': 'per-document-bernoulli-shard-seeded-v1', 'source': source,
              'input': item, 'index_format': '<u8 [source_row, output_offset, jsonl_bytes, text_bytes]'}
    if marker.exists():
        info = read_json(marker)
        if info['recipe'] != recipe:
            raise ValueError('Completed shard sampling recipe changed')
        verify_output(target, info['output'])
        verify_output(index, info['index'])
        return info
    folder.mkdir(parents=True, exist_ok=True)
    for p in (target, index, target.with_suffix('.jsonl.part'), index.with_suffix('.bin.part')):
        archive_partial(p)
    partial, index_partial = target.with_suffix('.jsonl.part'), index.with_suffix('.bin.part')
    url = url or (f"https://huggingface.co/datasets/{source['repo']}/resolve/"
                  f"{source['revision']}/{item['path']}")
    seed = int.from_bytes(hashlib.sha256(
        f"dclm-document-sample-v1:{source['sampling_seed']}:{item['path']}".encode()).digest(), 'big')
    rng = random.Random(seed)
    seen = selected = text_bytes = 0
    with tempfile.TemporaryDirectory(prefix='dclm-shard-') as temporary:
        compressed = Path(temporary) / 'input.zst'
        with urllib.request.urlopen(url, timeout=120) as response, compressed.open('wb') as out:
            while chunk := response.read(BLOCK):
                out.write(chunk)
        if compressed.stat().st_size != item['size'] or sha256_file(compressed) != item['sha256']:
            raise ValueError(f'Compressed DCLM shard checksum mismatch: {item["path"]}')
        with compressed.open('rb') as inp, zstandard.ZstdDecompressor().stream_reader(inp) as decoder, \
                io.BufferedReader(decoder, buffer_size=BLOCK) as lines, \
                partial.open('xb', buffering=BLOCK) as out, index_partial.open('xb', buffering=BLOCK) as idx:
            for source_row, line in enumerate(lines):
                if not line.strip():
                    continue
                seen += 1
                if rng.random() >= source['sampling_fraction']:
                    continue
                record = json.loads(line)
                if not isinstance(record.get('text'), str):
                    raise ValueError(f'Invalid text at {item["path"]}:{source_row}')
                encoded = line if line.endswith(b'\n') else line + b'\n'
                nbytes = len(record['text'].encode('utf-8'))
                idx.write(struct.pack('<QQQQ', source_row, out.tell(), len(encoded), nbytes))
                out.write(encoded)
                text_bytes += nbytes
                selected += 1
        info = {'recipe': recipe, 'source_records': seen, 'records': selected,
                'text_bytes': text_bytes, 'output': file_info(partial), 'index': file_info(index_partial),
                'output_path': str(target), 'index_path': str(index)}
        partial.replace(target)
        index_partial.replace(index)
        write_json(marker, info)  # only this last atomic write publishes a complete shard
        return info


def sample_shard_with_retry(folder, source, item, report_reuse=False):
    # A brief cluster DNS outage must not discard an hour-long scan allocation.
    # Keep the retries bounded; published shard hashes and sampling stay unchanged.
    attempts = 8
    marker = Path(folder) / (hashlib.sha256(item['path'].encode()).hexdigest() + '.json')
    reused = marker.exists()
    for attempt in range(attempts):
        try:
            part = sample_shard(folder, source, item)
            return (part, reused) if report_reuse else part
        except (OSError, EOFError, zstandard.ZstdError) as error:
            # Retry only transient I/O/decompression failures, never bad hashes or data.
            emit(stage='download', shard=item['path'], attempt=attempt + 1, error=type(error).__name__)
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))


def assemble_parent(folder, source, parts):
    label = f"{source['sampling_fraction'] * 100:g}pct"
    parent = folder / label / f'dclm_baseline_{label}.jsonl'
    index = parent.with_name('documents.index.bin')
    marker = folder / 'sample_provenance.json'
    recipe = {'source': source, 'parts_sha256': json_hash(parts), 'sampling_unit': 'document',
              'sampling_note': 'Independent Bernoulli selection, not an exact fixed-count 1% sample'}
    if marker.exists():
        info = read_json(marker)
        if info['recipe'] != recipe:
            raise ValueError('Parent DCLM recipe changed')
        verify_output(parent, info['output'])
        verify_output(index, info['index'])
        return info
    parent.parent.mkdir(parents=True, exist_ok=True)
    partial, index_partial = parent.with_suffix('.jsonl.part'), index.with_suffix('.bin.part')
    for p in (parent, index, partial, index_partial):
        archive_partial(p)
    records = sum(p['records'] for p in parts)
    with partial.open('xb', buffering=BLOCK) as out, index_partial.open('xb', buffering=BLOCK) as idx:
        for number, part in enumerate(parts, 1):
            offset = out.tell()
            data = np.fromfile(part['index_path'], dtype='<u8').reshape(-1, 4)
            if len(data) != part['records']:
                raise ValueError('DCLM shard index record count changed')
            parent_index = data[:, 1:].copy()
            parent_index[:, 0] += offset
            idx.write(parent_index.tobytes())
            with Path(part['output_path']).open('rb') as inp:
                h = hashlib.sha256()
                while chunk := inp.read(BLOCK):
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != part['output']['sha256']:
                raise ValueError('DCLM shard changed during parent assembly')
            if number % 100 == 0:
                emit(stage='assemble', shards_done=number, shards_total=len(parts))
    info = {'recipe': recipe, 'source_records': sum(p['source_records'] for p in parts),
            'records': records, 'text_bytes': sum(p['text_bytes'] for p in parts),
            'output': file_info(partial), 'index': file_info(index_partial),
            'output_path': str(parent), 'index_path': str(index),
            'index_format': '<u8 [offset, jsonl_bytes, text_bytes]'}
    partial.replace(parent)
    index_partial.replace(index)
    write_json(marker, info)
    emit(stage='parent_ready', records=records, text_bytes=info['text_bytes'],
         actual_document_fraction=records / info['source_records'])
    return info


def create_100gb_parent(folder, sample, target_bytes=100_000_000_000):
    """Random whole-document 100GB parent, drawn across the complete global sample."""
    folder = Path(folder)
    path = folder / '100GB/dclm_100gb.jsonl'
    index_path = path.with_name('documents.index.bin')
    marker = folder / 'provenance.json'
    seed = pool_seed('100GB', 42)
    recipe = {'algorithm': 'random-document-permutation-byte-budget-v1',
              'sample_sha256': sample['output']['sha256'], 'target_text_bytes': target_bytes,
              'seed': seed, 'source': sample['recipe']['source']}
    if marker.exists():
        info = read_json(marker)
        if info['recipe'] != recipe:
            raise ValueError('100GB DCLM parent recipe changed')
        verify_output(path, info['output'])
        verify_output(index_path, info['index'])
        return info
    index = np.memmap(sample['index_path'], mode='r', dtype='<u8', shape=(sample['records'], 3))
    chosen, actual = choose_indices(index, target_bytes, seed, set())
    selected = np.zeros(len(index), dtype=bool)
    selected[chosen] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    partial, index_partial = path.with_suffix('.jsonl.part'), index_path.with_suffix('.bin.part')
    for p in (path, index_path, partial, index_partial):
        archive_partial(p)
    h = hashlib.sha256()
    with Path(sample['output_path']).open('rb', buffering=BLOCK) as inp, \
            partial.open('xb', buffering=BLOCK) as out, index_partial.open('xb', buffering=BLOCK) as idx:
        row = -1
        for row, line in enumerate(inp):
            h.update(line)
            if len(line) != int(index[row, 1]):
                raise ValueError('Sample/index alignment changed')
            if selected[row]:
                idx.write(struct.pack('<QQQ', out.tell(), len(line), int(index[row, 2])))
                out.write(line)
            if (row + 1) % 100000 == 0:
                emit(stage='parent', documents_done=row + 1, documents_total=len(index))
    if row + 1 != len(index) or h.hexdigest() != sample['output']['sha256']:
        raise ValueError('DCLM sample changed while selecting 100GB parent')
    info = {'recipe': recipe, 'records': len(chosen), 'text_bytes': actual,
            'overshoot_bytes': actual - target_bytes, 'output': file_info(partial),
            'index': file_info(index_partial), 'output_path': str(path), 'index_path': str(index_path),
            'sample_manifest_sha256': sha256_file(folder / 'sample_provenance.json')}
    partial.replace(path)
    index_partial.replace(index_path)
    write_json(marker, info)
    emit(stage='parent_ready', records=len(chosen), text_bytes=actual, overshoot_bytes=actual-target_bytes)
    return info


def download(root, workers=8):
    folder = Path(root) / 'raw_text_dclm'
    source = read_json(SOURCE)
    with lock(root, 'dclm-download'):
        files = load_inventory(folder, source)
        emit(stage='download', shards_total=len(files), compressed_bytes_total=source['compressed_bytes'],
             workers=workers, sampling_fraction=source['sampling_fraction'])
        done = records = scanned = compressed_bytes = new_compressed_bytes = 0
        by_path = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(sample_shard_with_retry, folder / 'sampled_shards', source, f, True): f
                       for f in files}
            try:
                for future in as_completed(futures):
                    item = futures[future]
                    part, reused = future.result()
                    by_path[item['path']] = part
                    done += 1
                    records += part['records']
                    scanned += part['source_records']
                    compressed_bytes += item['size']
                    if not reused:
                        new_compressed_bytes += item['size']
                    emit(stage='download', shards_done=done, shards_total=len(files),
                         compressed_bytes_done=compressed_bytes, compressed_bytes_total=source['compressed_bytes'],
                         new_compressed_bytes_done=new_compressed_bytes,
                         records=records, source_records=scanned)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        # Stable source order, independent of worker completion order.
        sample = assemble_parent(folder, source, [by_path[f['path']] for f in files])
        return create_100gb_parent(folder, sample, source['parent_target_text_bytes'])


def choose_indices(index, target_bytes, seed, excluded):
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(index))
    order = order[~np.isin(order, list(excluded))]
    sizes = index[order, 2]
    cumulative = np.cumsum(sizes, dtype=np.uint64)
    if not len(cumulative) or cumulative[-1] < target_bytes:
        raise ValueError('The completed 1% sample is too small for the requested pool')
    count = int(np.searchsorted(cumulative, target_bytes, side='left')) + 1
    return np.sort(order[:count]), int(cumulative[count - 1])


def sample_pools(root, targets=None, heldout_count=1000):
    targets = POOLS if targets is None else targets
    folder = Path(root) / 'raw_text_dclm'
    with lock(root, 'dclm-pools'):
        parent = read_json(folder / 'provenance.json')
        path, index_path = Path(parent['output_path']), Path(parent['index_path'])
        verify_output(path, parent['output'])
        verify_output(index_path, parent['index'])
        index = np.memmap(index_path, mode='r', dtype='<u8', shape=(parent['records'], 3))
        excluded = set(random.Random(HELDOUT_SEED).sample(range(len(index)), heldout_count))
        heldout = folder / 'heldout' / f'dclm_100gb_shared_{heldout_count}.json'
        docs = []
        with path.open('rb') as inp:
            for row in sorted(excluded):
                offset, nbytes, _ = map(int, index[row])
                inp.seek(offset)
                docs.append({'index': row, 'text': json.loads(inp.read(nbytes))['text']})
        shared = {'source_sha256': parent['output']['sha256'], 'seed': HELDOUT_SEED, 'documents': docs}
        if heldout.exists() and read_json(heldout) != shared:
            raise ValueError('Existing DCLM heldout changed')
        if not heldout.exists():
            write_json(heldout, shared)
        outputs, selections, recipes, actual_bytes = {}, {}, {}, {}
        for size, budget in targets.items():
            target = folder / size / f'dclm_{size.lower()}.jsonl'
            recipe = {'algorithm': 'independent-random-document-permutation-byte-budget-v1',
                      'source_sha256': parent['output']['sha256'], 'target_text_bytes': budget,
                      'seed': pool_seed(size, 42), 'heldout_sha256': sha256_file(heldout),
                      'heldout_file': str(heldout), 'data_size': size}
            marker = target.with_name('pool.json')
            if marker.exists():
                info = read_json(marker)
                if info['recipe'] != recipe:
                    raise ValueError('DCLM pool recipe changed')
                verify_output(target, info['output'])
                continue
            chosen, actual = choose_indices(index, budget, recipe['seed'], excluded)
            selected = np.zeros(len(index), dtype=bool)
            selected[chosen] = True
            selections[size], recipes[size], actual_bytes[size] = selected, recipe, actual
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix('.jsonl.part')
            archive_partial(target)
            archive_partial(partial)
            outputs[size] = (target, partial)
        # One sequential scan for all independent pools; no random Weka reads.
        from contextlib import ExitStack
        h = hashlib.sha256()
        with ExitStack() as stack:
            handles = {size: stack.enter_context(partial.open('xb', buffering=BLOCK))
                       for size, (_, partial) in outputs.items()}
            if handles:
                with path.open('rb', buffering=BLOCK) as inp:
                    row = -1
                    for row, line in enumerate(inp):
                        h.update(line)
                        if len(line) != int(index[row, 1]):
                            raise ValueError('DCLM parent/index alignment changed')
                        for size, out in handles.items():
                            if selections[size][row]:
                                out.write(line)
                        if (row + 1) % 100000 == 0:
                            emit(stage='pools', documents_done=row + 1, documents_total=len(index))
                if row + 1 != len(index) or h.hexdigest() != parent['output']['sha256']:
                    raise ValueError('DCLM parent changed while sampling')
        for size, (target, partial) in outputs.items():
            info = {'recipe': recipes[size], 'records': int(selections[size].sum()),
                    'text_bytes': actual_bytes[size], 'overshoot_bytes': actual_bytes[size] - targets[size],
                    'output': file_info(partial), 'output_path': str(target), 'heldout_records': heldout_count}
            partial.replace(target)
            write_json(target.with_name('pool.json'), info)
            emit(stage='pool_ready', pool=size, records=info['records'], text_bytes=info['text_bytes'],
                 overshoot_bytes=info['overshoot_bytes'])
        return {size: read_json(folder / size / 'pool.json') for size in targets}


def tokenizer_path(root, pool, vocab):
    return Path(root) / 'tokenizers_dclm' / f'default_{pool}_{vocab}' / 'tokenizer.json'


def train_bpe(root, pool, vocab):
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tokenizers import Tokenizer
    from train_tokenizer import validate_tokenizer
    target = tokenizer_path(root, pool, vocab)
    raw = Path(root) / 'raw_text_dclm' / pool / f'dclm_{pool.lower()}.jsonl'
    pool_info = read_json(raw.with_name('pool.json'))
    with lock(root, f'dclm-tokenizer-{pool}-{vocab}'):
        verify_output(raw, pool_info['output'])
        signature = raw_signature(raw)
        recipe = {'pool': pool_info, 'vocab_size': vocab, 'pretokenizer': 'bytelevel',
                  'trainer': 'same train_tokenizer.py as Dolma', 'code_revision': os.environ.get('TOKENIZER_CODE_REVISION', 'local')}
        marker = target.with_name('pipeline.json')
        if marker.exists():
            info = read_json(marker)
            for k in ('pool', 'vocab_size', 'pretokenizer', 'trainer'):
                if info['recipe'][k] != recipe[k]:
                    raise ValueError('DCLM tokenizer recipe changed')
            verify_output(target, info['output'])
            validate_tokenizer(Tokenizer.from_file(str(target)), vocab, 'bytelevel')
            return info
        staging = target.with_name('tokenizer.staging.json')
        for p in (target, staging, staging.with_name(staging.name + '.part')):
            archive_partial(p)
        emit(stage='tokenizer', pool=pool, vocab_size=vocab, state='started')
        run([os.sys.executable, 'train_tokenizer.py', '--data', raw, '--output', staging,
             '--vocab-size', vocab, '--pretokenizer', 'bytelevel'])
        if raw_signature(raw) != signature or sha256_file(raw) != pool_info['output']['sha256']:
            raise ValueError('DCLM pool changed while training tokenizer')
        validate_tokenizer(Tokenizer.from_file(str(staging)), vocab, 'bytelevel')
        info = {'recipe': recipe, 'output': file_info(staging), 'output_path': str(target),
                'verified': {'vocab_size': vocab, 'eos_id': 0, 'all_256_bytes': True, 'round_trip': True}}
        staging.replace(target)
        write_json(marker, info)
        emit(stage='tokenizer_ready', pool=pool, vocab_size=vocab, sha256=info['output']['sha256'])
        return info


def tokenized_path(root, pool, vocab):
    return Path(root) / 'tokenized_data_dclm' / f'default_{pool}_{vocab}'


def verify_token_stream(directory, pool_info, vocab, tokenizer_sha):
    from pipeline import completed_meta
    meta = completed_meta(directory, vocab)
    if (meta['seed'] != 42 or meta['eval_fraction'] != .01 or meta['eos_token_id'] != 0
            or meta['data_files'] != [pool_info['output_path']]
            or sum(meta[s]['documents'] for s in ('train', 'eval')) != pool_info['records']
            or sum(meta[s]['bytes'] for s in ('train', 'eval')) != pool_info['text_bytes']
            or sha256_file(directory / 'tokenizer.json') != tokenizer_sha):
        raise ValueError('DCLM token stream corpus, split or tokenizer mismatch')
    repack = read_json(directory / 'repack.json')
    if repack['tokenizer_sha256'] != tokenizer_sha or repack['dtype'] != 'uint32':
        raise ValueError('DCLM token stream repack identity mismatch')
    for split in ('train', 'eval'):
        if (any(repack[split][k] != meta[split][k] for k in ('documents', 'bytes', 'tokens'))
                or sha256_file(directory / f'{split}.bin') != repack[split]['token_sha256']
                or sha256_file(directory / f'{split}.documents.csv.gz') != repack[split]['sha256']):
            raise ValueError(f'DCLM {split} token stream hash/count mismatch')
    return meta, repack


def tokenize_pool(root, pool, vocab):
    """Encode every pool document once; atomically publish verified uint32 streams."""
    raw = Path(root) / 'raw_text_dclm' / pool / f'dclm_{pool.lower()}.jsonl'
    target = tokenized_path(root, pool, vocab)
    tokenizer = tokenizer_path(root, pool, vocab)
    with lock(root, f'dclm-tokens-{pool}-{vocab}'):
        pool_info = read_json(raw.with_name('pool.json'))
        bpe = read_json(tokenizer.with_name('pipeline.json'))
        if bpe['recipe']['pool'] != pool_info or bpe['verified']['vocab_size'] != vocab:
            raise ValueError('DCLM tokenizer/pool identity mismatch')
        verify_output(raw, pool_info['output'])
        verify_output(tokenizer, bpe['output'])
        heldout = Path(pool_info['recipe']['heldout_file'])
        if sha256_file(heldout) != pool_info['recipe']['heldout_sha256']:
            raise ValueError('DCLM held-out exclusion changed')
        recipe = {'pool': pool_info, 'tokenizer_sha256': bpe['output']['sha256'],
                  'vocab_size': vocab, 'seed': 42, 'eval_fraction': .01,
                  'eos_per_document': 1, 'dtype': 'uint32'}
        marker = target / 'pipeline.json'
        if marker.exists():
            info = read_json(marker)
            if info['recipe'] != recipe:
                raise ValueError('Published DCLM tokenization recipe changed')
            meta, repack = verify_token_stream(target, pool_info, vocab, recipe['tokenizer_sha256'])
            if info['meta'] != meta or info['repack'] != repack:
                raise ValueError('Published DCLM tokenization metadata changed')
            return info
        staging = target.with_name(target.name + '.staging')
        archive_partial(target)
        archive_partial(staging)
        emit(stage='tokenization', pool=pool, vocab_size=vocab, state='started',
             documents_total=pool_info['records'], text_bytes_total=pool_info['text_bytes'])
        run([os.sys.executable, 'tokenize_data.py', '--data', raw, '--tokenizer', tokenizer,
             '--output-dir', staging, '--seed', '42', '--eval-fraction', '0.01', '--progress-json'])
        verify_output(raw, pool_info['output'])
        verify_output(tokenizer, bpe['output'])
        if read_json(raw.with_name('pool.json')) != pool_info or sha256_file(heldout) != recipe['pool']['recipe']['heldout_sha256']:
            raise ValueError('DCLM pool/exclusion changed during tokenization')
        meta, repack = verify_token_stream(staging, pool_info, vocab, recipe['tokenizer_sha256'])
        info = {'recipe': recipe, 'meta': meta, 'repack': repack, 'output_path': str(target),
                'total_tokens': sum(meta[s]['tokens'] for s in ('train', 'eval')),
                'verified': {'full_pool': True, 'token_stream_hashes': True,
                             'document_metadata_hashes': True, 'tokenizer_hash': True}}
        write_json(staging / 'pipeline.json', info)
        staging.replace(target)
        emit(stage='tokenization_ready', pool=pool, vocab_size=vocab,
             train_tokens=meta['train']['tokens'], eval_tokens=meta['eval']['tokens'], total_tokens=info['total_tokens'])
        return info


def run_stage(args):
    if args.stage == 'download-dclm':
        return download(args.root, args.workers)
    if args.stage == 'sample-dclm':
        return sample_pools(args.root)
    if args.stage == 'train-dclm-tokenizer':
        return train_bpe(args.root, args.data_size, args.vocab_size)
    if args.stage == 'tokenize-dclm-data':
        return tokenize_pool(args.root, args.data_size, args.vocab_size)
    raise ValueError('Only DCLM data and tokenizer preparation is authorized')


def summary(args):
    folder = args.root / 'raw_text_dclm'
    if args.stage == 'download-dclm':
        return {'data': read_json(folder / 'provenance.json')}
    if args.stage == 'sample-dclm':
        return {'pools': {p: read_json(folder / p / 'pool.json') for p in POOLS}}
    if args.stage == 'tokenize-dclm-data':
        info = read_json(tokenized_path(args.root, args.data_size, args.vocab_size) / 'pipeline.json')
        return {'meta': info['meta'], 'tokenization': info}
    return {'tokenizer': read_json(tokenizer_path(args.root, args.data_size, args.vocab_size).with_name('pipeline.json'))}
