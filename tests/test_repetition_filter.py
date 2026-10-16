"""Use real EMO datasets, collation and backward passes; no production jobs."""
import copy
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_core.data import DataCollator, NumpyFSLDataLoader, TokenizerConfig
from olmo_core.data.utils import get_labels
from scripts.repetition_filter import (FILTER_NAME, FILTER_SETTINGS, configure_repetition_filter,
                                       verify_instance_filter)
from scripts.training_callbacks import ReportMetricsCallback
from scripts.training_recipe import checkpoint_recipe_guard, LEGACY_RECIPE
from train import training_dataset, train_module_config
from test_pipeline import make_meta_fixture, pipeline, submit
from test_training_recipe import small_config, tiny_model


def dataset(path, *, mode='fsl', enabled=True, seq=512, documents=4):
    return training_dataset(path, {'train': {'documents': documents}}, data_loader=mode,
        repetition_filter=FILTER_NAME if enabled else 'none', sequence_length=seq,
        dtype='uint32', work_dir=str(path / 'work'),
        tokenizer=TokenizerConfig(vocab_size=512, eos_token_id=0, pad_token_id=0))


@pytest.mark.parametrize('period', range(1, 14))
@pytest.mark.parametrize('count,keep', [(31, True), (32, False)])
def test_exact_upstream_repetition_boundary_off_alignment(tmp_path, period, count, keep):
    # Unique prefix/suffix prevent accidental repetition at either boundary.
    tokens = list(range(200, 217)) + list(range(1, period + 1)) * count + list(range(300, 330))
    np.array(tokens, dtype=np.uint32).tofile(tmp_path / 'train.bin')
    item = dataset(tmp_path, seq=len(tokens), documents=1)[0]
    assert item['instance_mask'] is keep
    assert item['input_ids'].tolist() == tokens


def test_period_14_is_not_filtered(tmp_path):
    tokens = np.tile(np.arange(1, 15, dtype=np.uint32), 40)
    tokens.tofile(tmp_path / 'train.bin')
    assert dataset(tmp_path, seq=len(tokens), documents=1)[0]['instance_mask'] is True


@pytest.fixture
def corpus(tmp_path):
    clean = list(range(1, 129)) * 4  # Period 128 is outside the filter.
    rows = [clean.copy() for _ in range(4)]
    rows[0][17:17 + 13 * 32] = list(range(1, 14)) * 32
    rows[1][17:17 + 13 * 31] = list(range(1, 14)) * 31
    rows[2][25:57] = [7] * 32
    np.array(rows, dtype=np.uint32).tofile(tmp_path / 'train.bin')
    with gzip.open(tmp_path / 'train.documents.csv.gz', 'wt') as output:
        for i in range(4):
            output.write(f'{i * 512},{(i + 1) * 512}\n')
    return tmp_path


@pytest.mark.parametrize('mode,epoch', [('fsl', 1), ('repack', 1), ('repack', 2)])
def test_masks_do_not_change_tokens_order_length_or_labels_of_clean_instances(corpus, mode, epoch):
    original = (corpus / 'train.bin').read_bytes()
    plain, filtered = dataset(corpus, mode=mode, enabled=False), dataset(corpus, mode=mode)
    if mode == 'repack':
        for ds in (plain, filtered):
            ds.repack_for_epoch(epoch, seed=42)
    assert len(plain) == len(filtered) == 4
    items = [filtered[i] for i in range(4)]
    assert sum(not x['instance_mask'] for x in items) == 2
    for i, item in enumerate(items):
        assert 'instance_mask' not in plain[i]
        torch.testing.assert_close(item['input_ids'], plain[i]['input_ids'], rtol=0, atol=0)
    batch = DataCollator(pad_token_id=0)(items)
    labels = get_labels(batch)
    for i, item in enumerate(items):
        if item['instance_mask']:
            torch.testing.assert_close(labels[i, :-1], item['input_ids'][1:], rtol=0, atol=0)
            assert labels[i, -1] == -100
        else:
            assert torch.all(labels[i] == -100)
    assert (corpus / 'train.bin').read_bytes() == original


@pytest.mark.parametrize('workers', [0, 2])
def test_filter_survives_distributed_loader_and_resume(corpus, workers):
    def loader(seed=42):
        ds = dataset(corpus, mode='repack')
        return NumpyFSLDataLoader(ds, global_batch_size=2 * 512,
            collator=DataCollator(pad_token_id=0), work_dir=ds.work_dir,
            seed=seed, num_workers=workers, num_threads=1, dp_rank=1, dp_world_size=2)
    dl = loader()
    dl.reshuffle(2, in_memory=True)
    iterator = iter(dl)
    next(iterator)
    saved = copy.deepcopy(dl.state_dict())
    expected = list(iterator)
    resumed = loader(seed=123)
    resumed.load_state_dict(saved)
    resumed.reshuffle(resumed.epoch, in_memory=True)
    actual = list(resumed)
    assert len(actual) == len(expected) == 1
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left['input_ids'], right['input_ids'], rtol=0, atol=0)
        torch.testing.assert_close(left['instance_mask'], right['instance_mask'], rtol=0, atol=0)


@pytest.mark.parametrize('keep', [[False, True], [False, False]])
def test_real_backward_ignores_masked_sequences_and_all_masked_batches_are_finite(keep):
    config = small_config()
    config['seq_length'] = 64
    config['training']['micro_batch_size'] = 1
    mc = train_module_config(config, 1)
    mc.dp_config = None
    mc.compile_model = False
    mc.scheduler = None
    module = mc.build(tiny_model(), device=torch.device('cpu'))
    metrics = {}
    def record(name, value, **kwargs):
        metrics[(kwargs.get('namespace'), name)] = value.detach().clone()
    module._trainer = SimpleNamespace(record_metric=record,
        record_ce_loss=lambda value, **kwargs: record('CE', value))
    inputs = torch.stack([torch.ones(64, dtype=torch.long), torch.arange(64)])
    batch = {'input_ids': inputs, 'instance_mask': torch.tensor(keep)}
    labels = get_labels(batch)
    valid = labels != -100
    # Preserve EMO's normalization: masked instances count as zero-loss tokens.
    denominator = valid.sum() + (~batch['instance_mask']).sum() * 64
    logits = module.model(inputs).float()
    ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(),
        ignore_index=-100, reduction='sum') / denominator
    z = 1e-5 * logits.logsumexp(-1).square()[valid].sum() / denominator
    params = list(module.model.parameters())
    expected = torch.autograd.grad(ce + z, params)
    module.train_batch(batch)
    torch.testing.assert_close(module.optim.latest_loss, ce.detach())
    torch.testing.assert_close(metrics[(None, 'train/masked instances (%)')], torch.tensor(1 - sum(keep) / 2))
    for param, grad in zip(params, expected):
        assert torch.isfinite(param.grad).all()
        torch.testing.assert_close(param.grad, grad, rtol=2e-5, atol=2e-7)
        if not any(keep):
            assert torch.count_nonzero(param.grad) == 0


@pytest.mark.parametrize('recipe', ['emo-dense-v1', LEGACY_RECIPE])
def test_distinct_fingerprint_and_same_setting_in_all_chain_stages(tmp_path, recipe):
    args = make_meta_fixture(tmp_path, 32000, .5, 4_000_000)
    args.training_recipe = recipe
    old, old_path = pipeline.training_plan(args)
    assert 'repetition_filter' not in old['training']
    assert '--repetition-filter' not in submit.stage_arguments(args)
    args.repetition_filter = FILTER_NAME
    config, path = pipeline.training_plan(args)
    assert old_path != path and '_RepFilter_' in path.name
    assert config['training']['repetition_filter'] == FILTER_SETTINGS
    unfiltered = copy.deepcopy(config)
    del unfiltered['training']['repetition_filter']
    assert unfiltered == old
    for stage in pipeline.MODEL_STAGES:
        downstream = copy.deepcopy(args)
        downstream.stage = stage
        downstream.evals = list(pipeline.EVALS)
        parsed = submit.parser().parse_args(submit.stage_arguments(downstream))
        assert parsed.repetition_filter == FILTER_NAME
        assert pipeline.training_plan(parsed) == (config, path)


def test_resume_rejects_filter_changes_both_directions(tmp_path):
    plain = small_config()
    filtered = copy.deepcopy(plain)
    configure_repetition_filter(filtered, FILTER_NAME)
    for saved, changed in [(plain, filtered), (filtered, plain)]:
        (tmp_path / 'config.json').write_text(json.dumps(saved))
        checkpoint_recipe_guard(tmp_path, saved)
        with pytest.raises(ValueError, match='repetition filter mismatch'):
            checkpoint_recipe_guard(tmp_path, changed)
    with pytest.raises(ValueError, match='Cannot disable'):
        configure_repetition_filter(filtered, 'none')
    filtered['training']['repetition_filter']['repetition_max_count'] = 64
    with pytest.raises(ValueError, match='pinned'):
        configure_repetition_filter(filtered)


def test_actual_dataset_evidence_and_metric_persistence(corpus):
    cfg = small_config()
    configure_repetition_filter(cfg, FILTER_NAME)
    evidence = verify_instance_filter(dataset(corpus), cfg)
    assert evidence['instance_filter_config']['repetition_max_count'] == 32
    with pytest.raises(RuntimeError, match='differs'):
        verify_instance_filter(dataset(corpus, enabled=False), cfg)
    callback = ReportMetricsCallback(str(corpus), .25, .25)
    callback.log_metrics(10, {'train/masked instances (%)': .25, 'train/masked labels (%)': .26})
    result = json.loads((corpus / 'report_metrics.json').read_text())
    assert result['train/masked instances (%)'] == .25
    assert result['train/masked labels (%)'] == .26


def test_training_entrypoint_enables_filter_without_filtering_validation(tmp_path, monkeypatch):
    import train
    args = make_meta_fixture(tmp_path, 32000, .5, 4_000_000)
    cfg, output = pipeline.training_plan(args)
    meta_path = pipeline.tokenized_dir(args) / 'meta.json'
    meta = pipeline.read_json(meta_path)
    meta.update(eos_token_id=0, eos_token='<eos>')
    pipeline.write_json(meta_path, meta)
    config_path = tmp_path / 'launch.json'
    pipeline.write_json(config_path, cfg)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(config_path),
        '--data-dir', str(pipeline.tokenized_dir(args)), '--output-dir', str(output),
        '--repetition-filter', FILTER_NAME])
    monkeypatch.setenv('WORLD_SIZE', '8')
    monkeypatch.setattr(train, 'prepare_training_environment', lambda **_: None)
    monkeypatch.setattr(train, 'teardown_training_environment', lambda: None)
    monkeypatch.setattr(train.TransformerConfig, 'build', lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(train.TransformerTrainModuleConfig, 'build',
                        lambda *a, **kw: SimpleNamespace(dp_process_group=None))
    captured = {}
    def build_loader(self, ds, **kwargs):
        captured['dataset'] = ds
        return SimpleNamespace()
    monkeypatch.setattr(train.NumpyDataLoaderConfig, 'build', build_loader)
    class BuildBoundary(Exception):
        pass
    def build_trainer(self, *args, **kwargs):
        captured['trainer_config'] = self
        raise BuildBoundary()  # No GPU allocation or training.
    monkeypatch.setattr(train.TrainerConfig, 'build', build_trainer)
    precision = torch.get_float32_matmul_precision()
    try:
        with pytest.raises(BuildBoundary):
            train.main()
    finally:
        torch.set_float32_matmul_precision(precision)
    assert captured['dataset'].instance_filter_config.repetition_max_count == 32
    callbacks = captured['trainer_config'].callbacks
    assert callbacks['lm_evaluator'].eval_dataset.instance_filter_config is None
    assert callbacks['training_recipe'].config['training']['repetition_filter'] == FILTER_SETTINGS
    assert callbacks['training_recipe'].dataset is captured['dataset']
