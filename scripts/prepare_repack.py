"""Recover exact document spans for existing tokenization, without changing tokens.

Run on a CPU machine with access to the raw pool and tokenized directory:
    python scripts/prepare_repack.py --data-dir /path/to/tokenized_data/default_5GB_32000

Replays the saved tokenizer/split and compares every emitted token to the existing
train/eval files. This is deliberately not an EOS scan: raw text can contain EOS.
Incomplete .part metadata is preserved on failure; published data is never edited.
"""
import argparse
from contextlib import ExitStack
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def publish_metadata(data_dir, stats, token_hashes):
    """Publish closed CSV .part files, then a small completion/provenance manifest."""
    manifest = {'format': 'tokenizer-repack-v1', 'dtype': 'uint32',
                'tokenizer_sha256': sha256(data_dir / 'tokenizer.json')}
    for split in ('train', 'eval'):
        name = f'{split}.documents.csv.gz'
        path = data_dir / name
        (data_dir / (name + '.part')).replace(path)
        manifest[split] = {'file': name, 'sha256': sha256(path), **stats[split],
                           'token_sha256': token_hashes[split].hexdigest()}
    partial = data_dir / 'repack.json.part'
    partial.write_text(json.dumps(manifest, indent=2) + '\n')
    partial.replace(data_dir / 'repack.json')
    return manifest


def prepare(data_dir, batch_size=256):
    from tokenizers import Tokenizer
    from tokenize_data import tokenized_documents

    data_dir = Path(data_dir)
    with (data_dir / '.repack.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (data_dir / 'repack.json').exists():
            raise FileExistsError(f'{data_dir}/repack.json already exists; refusing to overwrite it')
        for split in ('train', 'eval'):
            for suffix in ('', '.part'):
                path = data_dir / f'{split}.documents.csv.gz{suffix}'
                if path.exists():
                    raise FileExistsError(f'Preserved metadata needs inspection before retry: {path}')
        meta = json.loads((data_dir / 'meta.json').read_text())
        tokenizer = Tokenizer.from_file(str(data_dir / 'tokenizer.json'))
        stats = {split: {'documents': 0, 'bytes': 0, 'tokens': 0} for split in ('train', 'eval')}
        hashes = {split: hashlib.sha256() for split in stats}
        with ExitStack() as stack:
            sources = {split: stack.enter_context((data_dir / f'{split}.bin').open('rb')) for split in stats}
            outputs = {split: stack.enter_context(gzip.open(data_dir / f'{split}.documents.csv.gz.part', 'wt'))
                       for split in stats}
            for split, text, ids in tokenized_documents(
                    [Path(p) for p in meta['data_files']], tokenizer, meta['eos_token_id'],
                    meta['eval_fraction'], meta['seed'], batch_size):
                tokens = np.asarray(ids, dtype=np.uint32).tobytes()
                if sources[split].read(len(tokens)) != tokens:
                    raise ValueError(f'{split} token mismatch at document {stats[split]["documents"]}; '
                                     'no metadata published')
                start = stats[split]['tokens']
                outputs[split].write(f'{start},{start + len(ids)}\n')
                hashes[split].update(tokens)
                stats[split]['documents'] += 1
                stats[split]['bytes'] += len(text.encode('utf-8'))
                stats[split]['tokens'] += len(ids)
                done = sum(s['documents'] for s in stats.values())
                if done % 100000 == 0:
                    print(f'{done:,} documents verified', flush=True)
            for split in stats:
                if sources[split].read(1) or stats[split] != {k: meta[split][k] for k in stats[split]}:
                    raise ValueError(f'{split} split counts/content differ from meta.json; no metadata published')
        return publish_metadata(data_dir, stats, hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir), indent=2))
