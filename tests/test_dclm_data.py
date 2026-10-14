"""Real tiny compressed shards: sampling, restart integrity, byte pools and launch scope."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dclm_data as d
import dclm_campaign as controller
import pipeline
import submit
from campaign import ready_nodes, placement_arguments


def sampled_fixture(root):
    folder = root / 'raw_text_dclm'
    source = {'repo': 'fixture', 'revision': 'fixed', 'sampling_fraction': .5, 'sampling_seed': 42}
    parts = []
    for shard in range(3):
        compressed = root / f'{shard}.zst'
        raw = b''.join((json.dumps({'id': shard * 1000 + i, 'text': f'doc {i} café 한글 ' * (1+i%7)}, ensure_ascii=False)+'\n').encode()
                       for i in range(1000))
        compressed.write_bytes(zstandard.ZstdCompressor().compress(raw))
        item = {'path': f'shard-{shard}.jsonl.zst', 'size': compressed.stat().st_size,
                'blob_id': str(shard), 'sha256': d.sha256_file(compressed)}
        part = d.sample_shard(folder / 'sampled_shards', source, item, compressed.as_uri())
        assert d.sample_shard(folder / 'sampled_shards', source, item, compressed.as_uri()) == part
        parts.append(part)
    sample = d.assemble_parent(folder, source, parts)
    parent = d.create_100gb_parent(folder, sample, target_bytes=50_000)
    return sample, parent, parts


def test_document_sampling_parent_and_independent_measured_pools(tmp_path):
    sample, parent, parts = sampled_fixture(tmp_path)
    assert sample['source_records'] == 3000
    assert 1300 < sample['records'] < 1700
    index = np.fromfile(sample['index_path'], dtype='<u8').reshape(-1, 3)
    lines = Path(sample['output_path']).read_bytes().splitlines(keepends=True)
    assert len(index) == len(lines) == sample['records']
    assert list(index[:, 1]) == list(map(len, lines))
    assert int(index[:, 2].sum()) == sum(len(json.loads(x)['text'].encode()) for x in lines)
    assert parent['text_bytes'] >= 50_000
    assert parent['overshoot_bytes'] < max(index[:, 2])
    targets = {'5GB': 5000, '10GB': 10000, '20GB': 20000}
    pools = d.sample_pools(tmp_path, targets, heldout_count=10)
    assert d.sample_pools(tmp_path, targets, heldout_count=10) == pools
    selections = {}
    parent_docs = [json.loads(x) for x in Path(parent['output_path']).read_bytes().splitlines()]
    shared = d.read_json(Path(pools['5GB']['recipe']['heldout_file']))
    heldout_ids = {parent_docs[x['index']]['id'] for x in shared['documents']}
    for name, info in pools.items():
        docs = [json.loads(x) for x in Path(info['output_path']).read_bytes().splitlines()]
        selections[name] = {x['id'] for x in docs}
        assert len(docs) == len(selections[name]) == info['records']
        assert not heldout_ids & selections[name]
        assert info['text_bytes'] == sum(len(x['text'].encode()) for x in docs)
        assert targets[name] <= info['text_bytes'] < targets[name] + max(index[:, 2])
    assert selections['5GB'] - selections['10GB']
    assert selections['5GB'] & selections['10GB']
    # Finished parent/pools do not silently accept content changes.
    Path(pools['5GB']['output_path']).write_bytes(b'{"text":"changed"}\n')
    with pytest.raises(ValueError, match='changed'):
        d.sample_pools(tmp_path, targets, heldout_count=10)


def test_input_checksum_and_incomplete_outputs_are_preserved(tmp_path):
    source = {'repo': 'fixture', 'revision': 'fixed', 'sampling_fraction': 1., 'sampling_seed': 42}
    path = tmp_path / 'input.zst'
    path.write_bytes(zstandard.ZstdCompressor().compress(b'{"text":"hello"}\n'))
    item = {'path': 'a.zst', 'size': path.stat().st_size, 'sha256': '0'*64}
    with pytest.raises(ValueError, match='checksum'):
        d.sample_shard(tmp_path / 'parts', source, item, path.as_uri())
    item['sha256'] = d.sha256_file(path)
    identity = d.hashlib.sha256(item['path'].encode()).hexdigest()
    partial = tmp_path / 'parts' / f'{identity}.jsonl.part'
    partial.write_text('preserve partial evidence')
    part = d.sample_shard(tmp_path / 'parts', source, item, path.as_uri())
    assert part['records'] == 1
    assert any(p.read_text() == 'preserve partial evidence' for p in (tmp_path/'parts/partial_history').iterdir())
    changed = {**source, 'sampling_seed': 43}
    with pytest.raises(ValueError, match='recipe'):
        d.sample_shard(tmp_path / 'parts', changed, item, path.as_uri())


def test_too_small_sample_cannot_release_pools(tmp_path):
    sample, parent, parts = sampled_fixture(tmp_path)
    idx = np.fromfile(sample['index_path'], dtype='<u8').reshape(-1, 3)
    with pytest.raises(ValueError, match='too small'):
        d.choose_indices(idx, sample['text_bytes'] + 1, 42, set())


def test_cpu_only_dependency_graph_and_no_models():
    state = controller.make_state()
    controller.validate_scope(state)
    assert len(state['nodes']) == 11
    assert ready_nodes(state) == ['dclm-download']
    state['nodes']['dclm-download']['status'] = 'succeeded'
    assert ready_nodes(state) == ['dclm-pools']
    state['nodes']['dclm-pools']['status'] = 'succeeded'
    assert len(ready_nodes(state)) == 9
    for node in state['nodes'].values():
        args = submit.parser().parse_args(node['arguments'] + placement_arguments(node))
        pipeline.validate_args(args)
        task = submit.make_spec(args)['tasks'][0]
        assert task['resources']['gpuCount'] == 0
        assert task['constraints']['cluster'] == ['ai2/saturn', 'ai2/neptune']
        assert {'name': 'CUDA_VISIBLE_DEVICES', 'value': ''} in task['envVars']
        assert task['context']['minRuntime'] == ('1h' if node['data_size'] == '5GB' else '3h')
        assert not node.get('chain_evaluation')
        assert task['datasets'][1]['source'] == {'weka': 'oe-training-default'}
    state['nodes']['dclm-download']['stage'] = 'train-model'
    with pytest.raises(ValueError, match='Unauthorized'):
        controller.validate_scope(state)


def test_summary_dispatch_has_no_name_collision(tmp_path):
    folder = tmp_path / 'raw_text_dclm'
    folder.mkdir()
    pipeline.write_json(folder/'provenance.json', {'records': 5})
    args = submit.parser().parse_args(['download-dclm', '--root', str(tmp_path)])
    assert pipeline.publish_summary(args)['data']['records'] == 5


def test_real_bpe_training_reuses_verified_pool_and_preserves_dolma_paths(tmp_path):
    sampled_fixture(tmp_path)
    d.sample_pools(tmp_path, {'5GB': 20000}, heldout_count=10)
    info = d.train_bpe(tmp_path, '5GB', 300)
    assert d.train_bpe(tmp_path, '5GB', 300) == info
    assert info['verified']['round_trip'] and info['verified']['vocab_size'] == 300
    assert Path(info['output_path']).is_file()
    assert not (tmp_path/'tokenizers').exists()
    assert not (tmp_path/'raw_text').exists()
    assert not (tmp_path/'models').exists()
