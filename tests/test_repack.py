"""First-epoch equivalence, upstream repack semantics, and checkpoint continuation."""
import copy
import gzip
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_core.data import DataCollator, NumpyFSLDataLoader, TokenizerConfig
from olmo_core.data.repack import NumpyFSLDynamicRepackedDataset
from olmo_core.data.utils import get_rng
from olmo_core.exceptions import OLMoConfigurationError
from scripts.prepare_repack import prepare, sha256
from train import training_dataset


@pytest.fixture
def corpus(tmp_path):
    # Documents deliberately include EOS internally and span sequence boundaries.
    lengths = [3, 8, 1, 6, 12, 2, 5, 7, 4, 9, 15, 11, 2, 5]
    documents, cursor = [], 1
    for length in lengths:
        documents.append(list(range(cursor, cursor + length - 1)) + [0])
        cursor += length
    documents[1][2] = 0
    tokens = np.array([t for d in documents for t in d], dtype=np.uint32)
    tokens.tofile(tmp_path / 'train.bin')
    with gzip.open(tmp_path / 'train.documents.csv.gz', 'wt') as output:
        pos = 0
        for document in documents:
            output.write(f'{pos},{pos + len(document)}\n')
            pos += len(document)
    return tmp_path, documents


def dataset(corpus, mode='repack', sequence_length=4):
    path, documents = corpus
    return training_dataset(
        path, {'train': {'documents': len(documents)}}, data_loader=mode,
        sequence_length=sequence_length, dtype='uint32', work_dir=str(path / 'data'),
        tokenizer=TokenizerConfig(vocab_size=256, eos_token_id=0, pad_token_id=0))


def loader(ds, rank=0, world_size=1, workers=0, seed=17):
    return NumpyFSLDataLoader(
        ds, global_batch_size=16, collator=DataCollator(pad_token_id=0),
        work_dir=ds.work_dir, seed=seed, num_workers=workers, num_threads=2,
        dp_rank=rank, dp_world_size=world_size)


def batches(dl):
    return [batch['input_ids'].tolist() for batch in dl]


@pytest.mark.parametrize('workers', [0, 2])
def test_epoch_one_default_matches_fsl_on_each_rank(corpus, workers):
    old_ds, new_ds = dataset(corpus, 'fsl'), dataset(corpus)
    assert isinstance(new_ds, NumpyFSLDynamicRepackedDataset)
    assert old_ds.fingerprint == new_ds.fingerprint
    # Epoch 1 does not require document sidecars, exactly like existing runs.
    (corpus[0] / 'train.documents.csv.gz').unlink()
    for rank in range(2):
        old = loader(old_ds, rank, 2, workers)
        new = loader(new_ds, rank, 2, workers)
        old.reshuffle(1)
        new.reshuffle(1)
        assert old.total_batches == new.total_batches
        assert batches(old) == batches(new)
    with pytest.raises(FileNotFoundError, match='prepare_repack.py'):
        new.reshuffle(2)


def test_epoch_two_matches_upstream_permutation_and_tail_policy(corpus):
    ds = dataset(corpus)
    documents = corpus[1]
    flat = [t for d in documents for t in d]
    for epoch in (1, 2, 3, 2):
        ds.repack_for_epoch(epoch, seed=17)
        order = np.arange(len(documents), dtype=np.uint32)
        if epoch > 1:
            get_rng(17 + epoch + 0x44594E52).shuffle(order)
        expected = [t for index in order for t in documents[index]]
        actual = torch.cat([ds[i]['input_ids'] for i in range(len(ds))]).tolist()
        assert actual == expected[:len(ds) * ds.sequence_length]
        assert len(ds) == len(flat) // ds.sequence_length
        if epoch > 1:
            assert actual != flat[:len(actual)]
            assert sorted(expected) == sorted(flat)
        assert torch.equal(ds[-1]['input_ids'], ds[len(ds) - 1]['input_ids'])
        with pytest.raises(IndexError):
            ds[len(ds)]


@pytest.mark.parametrize('workers', [0, 2])
def test_repacked_distributed_batches_and_resume(corpus, workers):
    # Reassemble both ranks in the loader's interleaved order: every global
    # instance appears once, and worker prefetching does not change ordering.
    whole = loader(dataset(corpus))
    whole.reshuffle(2, in_memory=True)
    expected = batches(whole)
    shards = []
    for rank in range(2):
        dl = loader(dataset(corpus), rank, 2, workers)
        dl.reshuffle(2, in_memory=True)
        shards.append(batches(dl))
        dl.reset()
        dl.reshuffle(2, in_memory=True)
        iterator = iter(dl)
        next(iterator)
        next(iterator)
        state = copy.deepcopy(dl.state_dict())
        remaining = [b['input_ids'].tolist() for b in iterator]
        resumed = loader(dataset(corpus), rank, 2, workers, seed=999)
        resumed.load_state_dict(state)  # restores checkpoint seed
        resumed.reshuffle(in_memory=True, epoch=resumed.epoch)
        assert batches(resumed) == remaining == shards[-1][2:]
    actual = [[row for pair in zip(a, b) for row in pair] for a, b in zip(*shards)]
    assert actual == expected
    assert len(actual) * 16 == sum(map(len, corpus[1])) // 16 * 16


def test_resume_old_epoch_one_checkpoint_then_repack(corpus):
    old = loader(dataset(corpus, 'fsl'))
    old.reshuffle(1, in_memory=True)
    iterator = iter(old)
    next(iterator)
    state = copy.deepcopy(old.state_dict())
    remaining = [batch['input_ids'].tolist() for batch in iterator]
    new = loader(dataset(corpus))
    new.load_state_dict(state)
    new.reshuffle(new.epoch, in_memory=True)
    assert batches(new) == remaining
    new.reset()
    new.reshuffle(2, in_memory=True)
    fresh = loader(dataset(corpus))
    fresh.reshuffle(2, in_memory=True)
    assert batches(new) == batches(fresh)


def test_later_epoch_resume_rejects_loader_or_metadata_changes(corpus):
    dl = loader(dataset(corpus))
    dl.reshuffle(2, in_memory=True)
    state = dl.state_dict()
    with pytest.raises(RuntimeError, match='ordinary FSL'):
        loader(dataset(corpus, 'fsl')).load_state_dict(state)
    missing = copy.deepcopy(state)
    missing.pop('repacking')
    with pytest.raises(RuntimeError, match='different repack plan'):
        loader(dataset(corpus)).load_state_dict(missing)
    # Same token contents/count, different document boundaries must not resume.
    path = corpus[0] / 'train.documents.csv.gz'
    with gzip.open(path, 'rt') as source:
        lines = source.readlines()
    lines[0], lines[1] = '0,2\n', '2,11\n'
    with gzip.open(path, 'wt') as output:
        output.writelines(lines)
    with pytest.raises(RuntimeError, match='different repack plan'):
        loader(dataset(corpus)).load_state_dict(state)


def test_corrupt_metadata_fails_before_repacking(corpus):
    path = corpus[0] / 'train.documents.csv.gz'
    (corpus[0] / 'repack.json').write_text(json.dumps({'train': {'sha256': 'bad'}}))
    with pytest.raises(OLMoConfigurationError, match='checksum mismatch'):
        dataset(corpus).repack_for_epoch(2, seed=17)
    (corpus[0] / 'repack.json').unlink()
    with gzip.open(path, 'wt') as output:
        output.write('0,100000\n')
    with pytest.raises(OLMoConfigurationError, match='invalid repack document span'):
        dataset(corpus).repack_for_epoch(2, seed=17)


def test_tokenization_sidecars_and_exact_legacy_recovery(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from train_tokenizer import EOS

    root = Path(__file__).resolve().parents[1]
    raw = tmp_path / 'raw.jsonl'
    texts = ['hello world', f'hello {EOS} world', '', 'hello'] * 50
    raw.write_text(''.join(json.dumps({'text': text}) + '\n' for text in texts))
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, EOS: 1, 'hello': 2, 'world': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens([EOS])
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    data = tmp_path / 'data'
    subprocess.run([sys.executable, str(root / 'tokenize_data.py'), '--data', str(raw),
                    '--tokenizer', str(tmp_path / 'tokenizer.json'), '--output-dir', str(data),
                    '--eval-fraction', '.2', '--seed', '42'], check=True, capture_output=True)
    manifest = json.loads((data / 'repack.json').read_text())
    boundaries = {}
    # Independently reproduce the original tokenizer's one-document split/order.
    rng = random.Random(42)
    expected = {'train': [], 'eval': []}
    for text in texts:
        split = 'eval' if rng.random() < .2 else 'train'
        expected[split].extend(tokenizer.encode(text, add_special_tokens=False).ids + [1])
    for split in expected:
        assert np.fromfile(data / f'{split}.bin', dtype=np.uint32).tolist() == expected[split]
        assert manifest[split]['token_sha256'] == sha256(data / f'{split}.bin')
        with gzip.open(data / f'{split}.documents.csv.gz', 'rt') as source:
            boundaries[split] = source.read()
    # Recovery writes only sidecars; all original artifacts remain byte-identical.
    original = {name: sha256(data / name) for name in ('meta.json', 'train.bin', 'eval.bin', 'tokenizer.json')}
    legacy = tmp_path / 'legacy'
    shutil.copytree(data, legacy)
    for name in ('train.documents.csv.gz', 'eval.documents.csv.gz', 'repack.json'):
        (legacy / name).unlink()
    prepare(legacy, batch_size=7)
    for name, checksum in original.items():
        assert sha256(legacy / name) == checksum
    for split in expected:
        with gzip.open(legacy / f'{split}.documents.csv.gz', 'rt') as source:
            assert source.read() == boundaries[split]
    with pytest.raises(FileExistsError):
        prepare(legacy)
    # Mismatched source leaves unpublished .part files for inspection.
    broken = tmp_path / 'broken'
    shutil.copytree(legacy, broken)
    for name in ('train.documents.csv.gz', 'eval.documents.csv.gz', 'repack.json'):
        (broken / name).unlink()
    with (broken / 'train.bin').open('r+b') as output:
        output.write(np.array([99], dtype=np.uint32).tobytes())
    with pytest.raises(ValueError, match='token mismatch'):
        prepare(broken)
    assert not (broken / 'repack.json').exists()
    assert (broken / 'train.documents.csv.gz.part').exists()
