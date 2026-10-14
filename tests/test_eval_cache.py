import json
from types import SimpleNamespace

import pytest
from datasets import Dataset, SplitDict, SplitInfo, Version
from datasets.arrow_writer import ArrowWriter

from eval import eval_suite as evaluator


def test_prepared_benchmark_loads_without_hub_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator.datasets_config, 'HF_DATASETS_CACHE', tmp_path)
    leaf = tmp_path / 'nyu-mll___blimp' / 'agreement' / '0.0.0' / 'fixture'
    leaf.mkdir(parents=True)
    data = Dataset.from_dict({'sentence_good': ['They run.'], 'sentence_bad': ['They runs.']})
    with ArrowWriter(path=str(leaf / 'blimp-train.arrow'), features=data.features) as writer:
        writer.write_table(data.data.table)
        writer.finalize()
    info = data.info
    info.dataset_name, info.config_name, info.builder_name = 'blimp', 'agreement', 'parquet'
    info.version = Version('0.0.0')
    info.splits = SplitDict({'train': SplitInfo(name='train', num_examples=1, num_bytes=data.data.nbytes)})
    info.write_to_directory(str(leaf))
    import httpx
    import requests
    def no_network(*args, **kwargs):
        raise AssertionError('Cached benchmark attempted an HTTP request')
    monkeypatch.setattr(httpx.Client, 'request', no_network)
    monkeypatch.setattr(requests.Session, 'request', no_network)
    before = evaluator.datasets_offline()
    restored = evaluator.load_eval_dataset('nyu-mll/blimp', 'agreement', split='train')
    assert restored.to_dict() == data.to_dict()
    assert evaluator.datasets_offline() == before


def test_complete_blimp_cache_skips_online_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator.datasets_config, 'HF_DATASETS_CACHE', tmp_path)
    names = [f'subset_{i:02d}' for i in range(67)]
    for name in names:
        p = tmp_path / 'nyu-mll___blimp' / name / '0.0.0' / 'hash' / 'dataset_info.json'
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'config_name': name}))
    monkeypatch.setattr(evaluator, 'get_dataset_config_names', lambda *a: pytest.fail('Hub lookup'))
    assert evaluator.blimp_configs() == names
    (p.parent / 'dataset_info.json').unlink()
    with evaluator.offline_datasets(), pytest.raises(ValueError, match='found 66'):
        evaluator.blimp_configs()


def test_cache_miss_rechecks_after_lock_then_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator.datasets_config, 'HF_DATASETS_CACHE', tmp_path)
    calls = []
    def load(*args, **kwargs):
        calls.append(evaluator.datasets_offline())
        if evaluator.datasets_offline():
            raise ValueError('No prepared cache')
        return 'complete-dataset'
    monkeypatch.setattr(evaluator, 'load_dataset', load)
    assert evaluator.load_eval_dataset('fixture/data') == 'complete-dataset'
    assert calls == [True, True, False]
    with evaluator.offline_datasets(), pytest.raises(ValueError, match='No prepared cache'):
        evaluator.load_eval_dataset('fixture/data')


def test_hub_throttle_respects_retry_after(monkeypatch):
    calls, waits = [], []
    def operation():
        calls.append(1)
        if len(calls) == 1:
            error = RuntimeError('rate limited')
            error.response = SimpleNamespace(status_code=429, headers={'Retry-After': '151'})
            raise error
        return 'complete'
    monkeypatch.setattr(evaluator.time, 'sleep', waits.append)
    assert evaluator.hub_retry(operation) == 'complete'
    assert waits == [153]
    with pytest.raises(ValueError, match='invalid'):
        evaluator.hub_retry(lambda: (_ for _ in ()).throw(ValueError('invalid')))
