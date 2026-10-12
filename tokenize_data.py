"""Tokenize raw text once and save it to a directory that train.py reads.

    python tokenize_data.py --data RAW --tokenizer tokenizer_32k.json --output-dir tokenized/32k

RAW is a .jsonl file (optionally .gz or .zst) or a directory of them. Every line
is a JSON object with a "text" field holding one document.

The output directory contains:
    train.bin, eval.bin   flat uint32 token streams; each document ends with EOS
    tokenizer.json        copy of the tokenizer
    meta.json             counts, tokens per byte, and the settings used

Documents are assigned to train or eval at random (default 1% eval, seed 42).
The assignment depends only on the seed and document order, so every tokenizer
gets the same split of the same raw data.
"""
import argparse
import json
import random
import shutil
from itertools import islice
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from train_tokenizer import EOS, data_files, iter_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='JSONL file or directory of JSONL files')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--eval-fraction', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=256, help='Documents tokenized per batch')
    args = parser.parse_args()

    if not 0 < args.eval_fraction < 1:
        parser.error('--eval-fraction must be between 0 and 1')
    if (args.output_dir / 'meta.json').exists():
        parser.error(f'{args.output_dir} is already complete')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos = tokenizer.token_to_id(EOS)
    if eos is None:
        raise ValueError(f'{EOS} must exist in the tokenizer')
    files = data_files(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    stats = {'train': {'documents': 0, 'bytes': 0, 'tokens': 0},
             'eval': {'documents': 0, 'bytes': 0, 'tokens': 0}}
    with (args.output_dir / 'train.bin').open('wb') as train, \
            (args.output_dir / 'eval.bin').open('wb') as evaluation:
        outputs = {'train': train, 'eval': evaluation}
        texts = iter_texts(files)
        while batch := list(islice(texts, args.batch_size)):
            for text, encoded in zip(batch, tokenizer.encode_batch(batch, add_special_tokens=False)):
                split = 'eval' if rng.random() < args.eval_fraction else 'train'
                np.asarray(encoded.ids + [eos], dtype=np.uint32).tofile(outputs[split])
                stats[split]['documents'] += 1
                stats[split]['bytes'] += len(text.encode('utf-8'))
                stats[split]['tokens'] += len(encoded.ids) + 1
            done = stats['train']['documents'] + stats['eval']['documents']
            if done % (args.batch_size * 1000) == 0:
                print(f'{done:,} documents', flush=True)

    total_bytes = stats['train']['bytes'] + stats['eval']['bytes']
    total_tokens = stats['train']['tokens'] + stats['eval']['tokens']
    meta = {
        'eos_token': EOS, 'eos_token_id': eos,
        'vocab_size': max(tokenizer.get_vocab().values()) + 1,
        'data_files': [str(path.resolve()) for path in files],
        'eval_fraction': args.eval_fraction, 'seed': args.seed,
        'tokens_per_byte': total_tokens / total_bytes,
        **stats,
    }
    shutil.copyfile(args.tokenizer, args.output_dir / 'tokenizer.json')
    # meta.json is written last and marks the directory as complete.
    meta_partial = args.output_dir / 'meta.json.part'
    meta_partial.write_text(json.dumps(meta, indent=2) + '\n')
    meta_partial.replace(args.output_dir / 'meta.json')
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
