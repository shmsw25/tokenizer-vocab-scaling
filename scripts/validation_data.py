"""Build one document-disjoint Dolma validation corpus and tokenizer-specific streams.

The 100M target is defined by the frozen 5GB/32K tokenizer, including EOS.
Whole documents are retained; every tokenizer evaluates exactly the same text.
"""
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

from pipeline import digest, lock, raw_signature, read_json, tokenized_dir, write_json

VERSION = 'dolma100m-v1'
TARGET_TOKENS = 100_000_000
SEED = 20261014


def validation_root(root):
    return Path(root) / 'validation' / VERSION


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prepare_corpus(root, *, target_tokens=TARGET_TOKENS, seed=SEED, candidate_count=None):
    from tokenizers import Tokenizer
    root = Path(root)
    folder = validation_root(root)
    marker, target = folder / 'manifest.json', folder / 'documents.jsonl'
    provenance = read_json(root / 'raw_text/provenance.json')
    full = Path(provenance['full']['path']); parent = Path(provenance['sample']['path'])
    reference = tokenized_dir(SimpleNamespace(root=root, data_size='5GB', vocab_size=32000)) / 'tokenizer.json'
    reference_meta = read_json(reference.with_name('meta.json'))
    if raw_signature(full) != provenance['full'] or raw_signature(parent) != provenance['sample']:
        raise ValueError('Downloaded corpus no longer matches provenance')
    if provenance['sampling_seed'] != 42 or provenance['sample_fraction'] != .4 or provenance['repetitions'] != 1:
        raise ValueError('Unsupported parent sampling recipe')
    heldout = read_json(root / 'raw_text/heldout/dolma3_100gb_shared_1000.json')
    recipe = dict(version=VERSION, algorithm='uniform-complement-shuffled-documents-v1',
        target_reference_tokens=target_tokens, seed=seed, full=provenance['full'],
        parent=provenance['sample'], reference_tokenizer=str(reference),
        reference_tokenizer_sha256=digest(reference), reference_pool='5GB', reference_vocab=32000,
        exclusion='Every record and exact UTF-8 text duplicate in the 100GB parent, including the 1000-document test set')
    with lock(root, 'validation-corpus'):
        if marker.exists():
            info = read_json(marker)
            if info['recipe'] != recipe or digest(target) != info['sha256']:
                raise ValueError('Existing validation corpus has changed or uses a different recipe')
            return info
        partial = target.with_suffix('.jsonl.part')
        if target.exists():
            raise FileExistsError(f'Unpublished validation corpus exists; inspect before retrying: {target}')
        if partial.exists():
            raise FileExistsError(f'Preserve and inspect partial validation before retrying: {partial}')
        folder.mkdir(parents=True, exist_ok=True)
        total = 0; full_hash = hashlib.sha256(); last = b''
        print('VALIDATION_PROGRESS counting the 250GB source', flush=True)
        with full.open('rb') as src:
            while chunk := src.read(8 * 1024 * 1024):
                total += chunk.count(b'\n'); full_hash.update(chunk); last = chunk[-1:]
        total += bool(last and last != b'\n')
        selected_count = int(total * .4)
        if selected_count != heldout['source_records']:
            raise ValueError('Parent record count differs from the original 40% sample')
        eligible = total - selected_count
        if candidate_count is None:
            mean_tokens = reference_meta['train']['tokens'] / reference_meta['train']['documents']
            candidate_count = min(eligible, math.ceil(3 * target_tokens / mean_tokens))
        if not 0 < candidate_count <= eligible:
            raise ValueError('Invalid validation candidate count')
        rng, candidate_rng = random.Random(42), random.Random(seed)
        needed, remaining, wanted = selected_count, eligible, candidate_count
        sampled_hash = hashlib.sha256(); candidates = {}; duplicates = 0
        print(f'VALIDATION_PROGRESS selecting {candidate_count:,} candidates outside the training parent', flush=True)
        with full.open('rb') as src:
            for index, line in enumerate(src):
                take_parent = needed and rng.randrange(total - index) < needed
                if take_parent:
                    sampled_hash.update(line if line.endswith(b'\n') else line + b'\n'); needed -= 1
                else:
                    take = wanted and candidate_rng.randrange(remaining) < wanted
                    remaining -= 1
                    if take:
                        text = json.loads(line)['text']; h = text_hash(text)
                        if h in candidates: duplicates += 1
                        else: candidates[h] = {'source_index': index, 'text': text, 'text_sha256': h}
                        wanted -= 1
                if index and index % 2_000_000 == 0:
                    print(f'VALIDATION_PROGRESS scanned {index:,}/{total:,} source records', flush=True)
        if needed or remaining or wanted or sampled_hash.hexdigest() != heldout['source_sha256']:
            raise ValueError('Reconstructed parent does not match the published training parent hash')
        # Index disjointness alone is insufficient when duplicate text occurs at different indices.
        parent_hash = hashlib.sha256(); overlap = 0
        print('VALIDATION_PROGRESS checking candidate text against the complete 100GB parent', flush=True)
        with parent.open('rb') as src:
            for line in src:
                parent_hash.update(line)
                if candidates.pop(text_hash(json.loads(line)['text']), None) is not None:
                    overlap += 1
        if parent_hash.hexdigest() != heldout['source_sha256']:
            raise ValueError('Parent content hash changed during validation exclusion')
        ordered = list(candidates.values()); random.Random(seed + 1).shuffle(ordered)
        tokenizer = Tokenizer.from_file(str(reference)); eos = tokenizer.token_to_id('<|endoftext|>')
        if eos is None: raise ValueError('Reference tokenizer is missing EOS')
        tokens = text_bytes = documents = 0; output_hash = hashlib.sha256()
        with partial.open('xb') as out:
            for start in range(0, len(ordered), 256):
                batch = ordered[start:start + 256]
                encoded = tokenizer.encode_batch([r['text'] for r in batch], add_special_tokens=False)
                for row, enc in zip(batch, encoded):
                    raw = (json.dumps(row, ensure_ascii=False) + '\n').encode()
                    out.write(raw); output_hash.update(raw)
                    tokens += len(enc.ids) + 1; text_bytes += len(row['text'].encode()); documents += 1
                    if tokens >= target_tokens: break
                if tokens >= target_tokens: break
        if tokens < target_tokens:
            raise ValueError(f'Insufficient disjoint candidates: {tokens} reference tokens; partial retained')
        if raw_signature(full) != provenance['full'] or raw_signature(parent) != provenance['sample']:
            raise ValueError('Source changed during validation preparation; partial retained')
        info = dict(recipe=recipe, documents=documents, text_bytes=text_bytes, reference_tokens=tokens,
            target_overshoot_tokens=tokens-target_tokens, sha256=output_hash.hexdigest(),
            full_sha256=full_hash.hexdigest(), parent_sha256=parent_hash.hexdigest(),
            candidate_records=candidate_count, candidate_text_duplicates=duplicates,
            candidate_parent_text_overlaps_removed=overlap, output=str(target),
            split_before_packing=True, training_parent_record_overlap=0, exact_text_overlap=0)
        partial.replace(target);write_json(marker, info)
        print('VALIDATION_READY ' + json.dumps(info), flush=True)
        return info


def tokenize_validation(root, pool, vocab):
    from tokenizers import Tokenizer
    import numpy as np
    root = Path(root);folder = validation_root(root)
    source = read_json(folder / 'manifest.json')
    raw = folder / 'documents.jsonl'
    tokenizer_path = root / 'tokenizers' / f'default_{pool}_{vocab}' / 'tokenizer.json'
    target = folder / 'tokenized' / f'default_{pool}_{vocab}'
    marker = target / 'meta.json'
    recipe = dict(validation_id=VERSION, documents_sha256=source['sha256'],
                  tokenizer_sha256=digest(tokenizer_path), pool=pool, vocab_size=vocab)
    with lock(root, f'validation-tokens-{pool}-{vocab}'):
        if marker.exists():
            meta = read_json(marker)
            if meta['recipe'] != recipe or (target/'eval.bin').stat().st_size != meta['tokens']*4 or digest(target/'eval.bin') != meta['sha256']:
                raise ValueError('Existing validation token stream does not match its manifest')
            return meta
        if digest(raw) != source['sha256']: raise ValueError('Validation corpus content changed')
        target.mkdir(parents=True, exist_ok=True);partial = target/'eval.bin.part'
        if (target/'eval.bin').exists():raise FileExistsError(f'Unpublished token stream exists: {target}')
        if partial.exists(): raise FileExistsError(f'Preserve partial validation tokenization: {partial}')
        tokenizer = Tokenizer.from_file(str(tokenizer_path));eos=tokenizer.token_to_id('<|endoftext|>')
        if eos is None or max(tokenizer.get_vocab().values())+1 != vocab:
            raise ValueError('Validation tokenizer identity mismatch')
        tokens=documents=text_bytes=0;h=hashlib.sha256()
        with raw.open() as src, partial.open('xb') as out:
            batch=[]
            def write_batch(rows):
                nonlocal tokens,documents,text_bytes
                encs=tokenizer.encode_batch([r['text'] for r in rows],add_special_tokens=False)
                for row,enc in zip(rows,encs):
                    ids=enc.ids+[eos];b=np.asarray(ids,dtype=np.uint32).tobytes()
                    out.write(b);h.update(b);tokens+=len(ids);documents+=1;text_bytes+=len(row['text'].encode())
            for line in src:
                batch.append(json.loads(line))
                if len(batch)==256: write_batch(batch);batch=[]
            if batch:write_batch(batch)
        if (documents,text_bytes)!=(source['documents'],source['text_bytes']):
            raise ValueError('Validation text statistics differ between vocabularies')
        meta=dict(recipe=recipe,tokens=tokens,documents=documents,bytes=text_bytes,sha256=h.hexdigest(),
                  tokens_per_byte=tokens/text_bytes,output=str(target/'eval.bin'),dtype='uint32')
        partial.replace(target/'eval.bin');write_json(marker,meta)
        print('VALIDATION_TOKENS_READY '+json.dumps(meta),flush=True)
        return meta


def prepare_validation(args):
    prepare_corpus(args.root)
    for pool in ('5GB','10GB','20GB'):
        for vocab in (32000,64000,128000):tokenize_validation(args.root,pool,vocab)


def validation_metadata(root, pool, vocab):
    """GPU jobs consume only an already-published CPU-prepared stream."""
    root=Path(root);folder=validation_root(root)
    manifest=read_json(folder/'manifest.json')
    target=folder/'tokenized'/f'default_{pool}_{vocab}'
    meta=read_json(target/'meta.json')
    tokenizer=root/'tokenizers'/f'default_{pool}_{vocab}'/'tokenizer.json'
    expected=dict(validation_id=VERSION,documents_sha256=manifest['sha256'],
        tokenizer_sha256=digest(tokenizer),pool=pool,vocab_size=vocab)
    if meta['recipe']!=expected or meta['bytes']!=manifest['text_bytes'] or meta['documents']!=manifest['documents']:
        raise ValueError('Validation stream corpus/tokenizer identity mismatch')
    if (target/'eval.bin').stat().st_size!=meta['tokens']*4 or digest(target/'eval.bin')!=meta['sha256']:
        raise ValueError('Validation token stream changed')
    return meta
