"""Independent sampling, shared holdout isolation, and pool-specific model paths."""
import json
from decimal import Decimal
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pipeline
from sample_pool import sample_pool
from test_pipeline import args_for, make_meta_fixture


def parent_fixture(root):
    source = root / 'raw_text/100GB/dolma3_100gb.jsonl'
    source.parent.mkdir(parents=True)
    source.write_text(''.join(json.dumps({'id': i, 'text': f'Doc {i} café 한국어 ' * (1 + i % 7)},
                                        ensure_ascii=False) + '\n' for i in range(10000)))
    pipeline.write_json(root / 'raw_text/provenance.json', {'sample': pipeline.raw_signature(source)})
    return source


def test_independent_pools_exact_counts_and_common_holdout(tmp_path):
    source = parent_fixture(tmp_path)
    selections, seeds = {}, set()
    heldout = None
    for size in pipeline.POOL_SIZES:
        info = sample_pool(tmp_path, size)
        target = Path(info['output']['path'])
        records = [json.loads(line) for line in target.read_text().splitlines()]
        ids = [doc['id'] for doc in records]
        assert len(ids) == int(Decimal(9000) * Decimal(size[:-2]) / 100)
        assert ids == sorted(set(ids))  # source order, no repetition
        assert info['text_bytes'] == sum(len(doc['text'].encode()) for doc in records)
        assert info['output_sha256'] == pipeline.digest(target)
        assert info['source_sha256'] == pipeline.digest(source)
        common = json.loads(Path(info['heldout_file']).read_text())
        excluded = {doc['index'] for doc in common['documents']}
        assert len(excluded) == 1000 and not set(ids) & excluded
        if heldout is None:
            heldout = common
        assert common == heldout
        seeds.add(info['sampling_seed'])
        selections[size] = set(ids)
        assert sample_pool(tmp_path, size) == info  # idempotent reuse, no rewrite
    assert len(seeds) == 6
    # Independent subsets overlap but are not nested.
    assert selections['5GB'] & selections['10GB']
    assert selections['5GB'] - selections['10GB']
    assert selections['2.5GB'] - selections['80GB']
    with pytest.raises(ValueError, match='different source or sampling recipe'):
        sample_pool(tmp_path, '5GB', base_seed=43)
    import eval.eval_suite as evaluator
    evaluator.DOLMA_HELDOUT_FILE = Path(info['heldout_file'])
    try:
        assert evaluator.dolma_sample() == heldout['documents']
    finally:
        evaluator.DOLMA_HELDOUT_FILE = None


def test_partial_pool_and_changed_parent_are_not_reused(tmp_path):
    source = parent_fixture(tmp_path)
    partial = tmp_path / 'raw_text/5GB/dolma3_5gb.jsonl.part'
    partial.parent.mkdir(parents=True)
    partial.write_text('preserve me')
    with pytest.raises(FileExistsError, match='Preserve and inspect'):
        sample_pool(tmp_path, '5GB')
    assert partial.read_text() == 'preserve me'
    source.write_text(source.read_text() + '{"text":"changed"}\n')
    with pytest.raises(ValueError, match='must finish'):
        sample_pool(tmp_path, '10GB')


def test_pool_paths_and_reference_are_specific_to_each_size(tmp_path):
    parent_fixture(tmp_path)
    outputs = set()
    for size in pipeline.TRAINING_POOLS:
        sample_pool(tmp_path, size)
        make_meta_fixture(tmp_path, 32000, 0.5, 4_000_000, size)
        args = make_meta_fixture(tmp_path, 64000, 0.4, 4_000_000, size)
        cfg, output = pipeline.training_plan(args)
        assert output.parent == tmp_path / 'models/small' / f'default_{size}_64000'
        assert cfg['seq_length'] == 3277 and cfg['training']['warmup_steps'] == 96
        assert cfg['pipeline']['pool']['data_size'] == size
        outputs.add(output)
        for stage in ('convert-model', 'evaluate-model'):
            downstream = args_for(stage, '--root', str(tmp_path), '--vocab-size', '64000', '--data-size', size)
            assert pipeline.training_plan(downstream)[1] == output
    assert len(outputs) == 3
