"""Bounded-memory, independent uniform sampling from the completed 100GB corpus."""
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
import random

from tqdm import tqdm

from pipeline import lock, raw_signature, read_json, write_json, digest

CHUNK_SIZE = 8 * 1024 * 1024
HELDOUT_SEED = 20261012
HELDOUT_DOCUMENTS = 1000


def pool_seed(size, base_seed):
    key = f'independent-uniform-pool-v1:{base_seed}:{size}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')


def sample_pool(root, size, base_seed=42, *, heldout_count=HELDOUT_DOCUMENTS):
    root = Path(root)
    fraction = Decimal(size[:-2]) / 100
    if not 0 < fraction <= 1:
        raise ValueError('Pool size must be greater than 0 and at most 100GB')
    source = root / 'raw_text/100GB/dolma3_100gb.jsonl'
    signature = raw_signature(source)
    if read_json(root / 'raw_text/provenance.json')['sample'] != signature:
        raise ValueError('The parent 100GB download/sample must finish before subsampling')
    target = root / 'raw_text' / size / f'dolma3_{size.lower()}.jsonl'
    if fraction == 1:
        target = root / 'raw_text/100GB/training/dolma3_100gb.jsonl'
    marker = target.with_name('pool.json')
    partial = target.with_name(target.name + '.part')
    heldout_file = root / 'raw_text/heldout' / f'dolma3_100gb_shared_{heldout_count}.json'
    seed = pool_seed(size, base_seed)
    recipe = {'algorithm': 'independent-uniform-without-replacement-v1',
              'source': signature, 'data_size': size, 'sample_fraction': str(fraction),
              'base_seed': base_seed, 'sampling_seed': seed,
              'heldout_seed': HELDOUT_SEED, 'heldout_documents': heldout_count,
              'heldout_file': str(heldout_file), 'repetitions': 1}
    with lock(root, f'raw-pool-{size}'):
        if target.exists():
            info = read_json(marker)
            if any(info.get(k) != v for k, v in recipe.items()):
                raise ValueError('Existing pool has a different source or sampling recipe')
            if info['output'] != raw_signature(target) or info['heldout_sha256'] != digest(heldout_file):
                raise ValueError('Published pool or held-out documents changed')
            print(f'Pool already complete: {target}', flush=True)
            return info
        if partial.exists():
            raise FileExistsError(f'Preserve and inspect incomplete sample before retrying: {partial}')
        target.parent.mkdir(parents=True, exist_ok=True)
        total, last_byte = 0, b''
        source_hash = hashlib.sha256()
        with source.open('rb') as stream, tqdm(total=signature['size'], desc='Counting records',
                unit='B', unit_scale=True, mininterval=5) as progress:
            while chunk := stream.read(CHUNK_SIZE):
                total += chunk.count(b'\n')
                last_byte = chunk[-1:]
                source_hash.update(chunk)
                progress.update(len(chunk))
        total += bool(last_byte and last_byte != b'\n')
        if raw_signature(source) != signature:
            raise ValueError('Parent corpus changed during counting')
        if total <= heldout_count:
            raise ValueError('Not enough documents for the shared holdout and training pool')
        excluded = set(random.Random(HELDOUT_SEED).sample(range(total), heldout_count))
        eligible = total - heldout_count
        selected_count = int(Decimal(eligible) * fraction)
        if selected_count == 0:
            raise ValueError('Requested sample contains no records')
        rng = random.Random(seed)
        needed, remaining = selected_count, eligible
        heldout, text_bytes, output_hash = [], 0, hashlib.sha256()
        print(f'Sampling {size}: {selected_count:,}/{eligible:,} eligible records; '
              f'independent seed {seed}; excluding {heldout_count} shared evaluation records', flush=True)
        with source.open('rb') as stream, partial.open('xb', buffering=CHUNK_SIZE) as output, tqdm(
                total=signature['size'], desc=f'Sampling {size}', unit='B', unit_scale=True,
                mininterval=5) as progress:
            for index, line in enumerate(stream):
                progress.update(len(line))
                if index in excluded:
                    heldout.append({'index': index, 'text': json.loads(line)['text']})
                    continue
                # Each size has its own RNG; conditional sampling yields exactly
                # k uniformly selected records, preserving their original order.
                take = needed and rng.randrange(remaining) < needed
                remaining -= 1
                if take:
                    text_bytes += len(json.loads(line)['text'].encode('utf-8'))
                    encoded = line if line.endswith(b'\n') else line + b'\n'
                    output.write(encoded)
                    output_hash.update(encoded)
                    needed -= 1
        if needed or remaining or len(heldout) != heldout_count or raw_signature(source) != signature:
            raise ValueError('Source changed or sample was incomplete; partial output retained')
        shared = {'source': signature, 'source_sha256': source_hash.hexdigest(),
                  'source_records': total, 'seed': HELDOUT_SEED, 'documents': heldout}
        heldout_file.parent.mkdir(parents=True, exist_ok=True)
        # Six jobs may finish together; serialize only the small manifest write.
        with heldout_file.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if heldout_file.exists():
                if read_json(heldout_file) != shared:
                    raise ValueError('Shared holdout does not match the parent corpus')
            else:
                write_json(heldout_file, shared)
        output_identity = raw_signature(partial)
        output_identity['path'] = str(target)
        info = {**recipe, 'source_sha256': source_hash.hexdigest(), 'source_records': total,
                'eligible_records': eligible, 'records': selected_count,
                'text_bytes': text_bytes, 'actual_text_gb': text_bytes / 1e9,
                'output': output_identity, 'output_sha256': output_hash.hexdigest(),
                'heldout_sha256': digest(heldout_file)}
        write_json(marker, info)
        partial.replace(target)
        print(f'Pool ready: {size}; {selected_count:,} records; {text_bytes / 1e9:.6f} GB text', flush=True)
        return info
