"""Offline recipe checks: no jobs, datasets, or production training runs."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pipeline
import submit
from loss_curve_health import invalid_result
from training_recipe import (EMO_RECIPE, EMO_SETTINGS, LEGACY_RECIPE,
                             campaign_arguments, checkpoint_recipe_guard, configure_recipe)
from train import TrainingRecipeCallback, optimizer_config, train_module_config, verify_live_recipe
from test_pipeline import make_meta_fixture
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelWrappingStrategy
from olmo_core.optim import AdamWConfig, SkipStepAdamWConfig


def small_config():
    return pipeline.read_json(pipeline.REPO / 'configs/small.json')


def tiny_model(*, tied=True, emo=True):
    return TransformerConfig.llama_like(
        d_model=16, n_layers=1, n_heads=2, vocab_size=128,
        tie_word_embeddings=tied, init_tied_output=emo,
        attn_backend=AttentionBackendName.torch).build(init_device='cpu')


@pytest.mark.parametrize('model_size', ['small', 'medium', 'medium_large'])
def test_production_builder_resolves_emo_recipe(model_size):
    config = pipeline.read_json(pipeline.REPO / f'configs/{model_size}.json')
    before = copy.deepcopy(config)
    module = train_module_config(config, 8)
    assert config['training_recipe'] == EMO_SETTINGS
    assert config['training'] == before['training']  # LR, WD, WSD and batch unchanged.
    assert isinstance(module.optim, SkipStepAdamWConfig)
    assert module.optim.betas == (.9, .95) and module.optim.eps == 1e-8
    assert module.optim.foreach and module.optim.step_increment_bugfix
    assert module.optim.rolling_interval_length == 128 and module.optim.sigma_factor == 6
    assert module.dp_config.name == DataParallelType.fsdp
    assert module.dp_config.param_dtype == DType.bfloat16
    assert module.dp_config.reduce_dtype == DType.float32
    assert module.dp_config.wrapping_strategy == TransformerDataParallelWrappingStrategy.full
    assert module.autocast_precision is None and module.z_loss_multiplier == 1e-5
    assert module.max_grad_norm == 1.0
    assert module.rank_microbatch_size == config['training']['micro_batch_size'] * 4096
    assert module.scheduler.warmup == 192 and module.scheduler.decay_fraction == .1
    assert config['model']['init_tied_output'] is True


def test_legacy_builder_preserves_config_and_optimizer():
    config = small_config()
    original = copy.deepcopy(config)
    module = train_module_config(config, 8, recipe=LEGACY_RECIPE)
    assert config == original
    assert isinstance(module.optim, AdamWConfig) and module.optim.fused
    assert module.optim.betas == (.9, .999)
    assert module.dp_config.name == DataParallelType.ddp
    assert module.autocast_precision == DType.bfloat16 and module.z_loss_multiplier is None


@pytest.mark.parametrize('kind', ['loss', 'gradient'])
def test_real_optimizer_skips_update_decay_moments_and_step_then_continues(kind):
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = optimizer_config({'learning_rate': .01, 'weight_decay': .1}).build(model)
    parameter = model.weight

    def step(loss, grad):
        parameter.grad = torch.full_like(parameter, grad)
        optimizer.latest_loss = torch.tensor(loss)
        optimizer.latest_grad_norm = torch.tensor(abs(grad))
        optimizer.step()

    for i in range(128):
        step(2.0 + .01 * (i % 2), .1 + .001 * (i % 2))
    before = parameter.detach().clone()
    state = copy.deepcopy(optimizer.state[parameter])
    step(100.0 if kind == 'loss' else 2.0, 100.0 if kind == 'gradient' else .1)
    assert optimizer.step_skipped.item() == 1
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    for key, value in state.items():
        torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
    step(2.0, .1)
    assert optimizer.step_skipped.item() == 0
    assert not torch.equal(parameter, before)
    assert optimizer.state[parameter]['step'].item() == state['step'].item() + 1


def test_real_train_module_backpropagates_z_loss_and_feeds_skip_optimizer():
    """Exercise EMO's real loss/backward/step path on synthetic CPU tensors.

    Only distributed wrapping, compilation and scheduling are omitted here;
    their production configuration is checked separately above.
    """
    config = small_config()
    config['seq_length'] = 8
    config['training']['micro_batch_size'] = 1
    module_config = train_module_config(config, 1)
    module_config.dp_config = None
    module_config.compile_model = False
    module_config.scheduler = None
    module = module_config.build(tiny_model(), device=torch.device('cpu'))
    metrics = {}

    def record(name, value, **kwargs):
        metrics[(kwargs.get('namespace'), name)] = value.detach().clone()

    module._trainer = SimpleNamespace(
        record_metric=record,
        record_ce_loss=lambda value, **kwargs: record('CE', value))
    inputs = torch.arange(16).reshape(2, 8)
    labels = (inputs + 1).clone()
    labels[:, -1] = -100
    valid = labels.flatten() != -100
    logits = module.model(inputs).reshape(-1, 128).float()
    ce = torch.nn.functional.cross_entropy(logits, labels.flatten(), ignore_index=-100)
    z = 1e-5 * logits.logsumexp(-1).square()[valid].mean()
    parameters = list(module.model.parameters())
    ce_gradients = torch.autograd.grad(ce, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(ce + z, parameters)
    assert any(not torch.equal(a, b) for a, b in zip(ce_gradients, expected_gradients))

    # A synthetic prior history makes this batch's real gradient norm an outlier.
    for i in range(128):
        module.optim.latest_loss = ce.detach() + .01 * (i % 2)
        module.optim.latest_grad_norm = torch.tensor(1e-7 + 1e-8 * (i % 2))
    module.train_batch({'input_ids': inputs, 'labels': labels})
    torch.testing.assert_close(metrics[('train', 'Z loss')], z.detach())
    torch.testing.assert_close(module.optim.latest_loss, ce.detach())
    for parameter, expected in zip(parameters, expected_gradients):
        torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-7)
    expected_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(parameter.grad) for parameter in parameters]))
    before = [parameter.detach().clone() for parameter in parameters]
    module.optim_step()
    torch.testing.assert_close(module.optim.latest_grad_norm, expected_norm)
    assert metrics[('optim', 'step skipped')].item() == 1
    for parameter, original in zip(parameters, before):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        state = module.optim.state[parameter]
        assert state['step'].item() == 0
        assert torch.count_nonzero(state['exp_avg']).item() == 0
        assert torch.count_nonzero(state['exp_avg_sq']).item() == 0
        assert state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32

    # With an ordinary prior history, the same real path updates normally.
    for i in range(128):
        module.optim.latest_loss = ce.detach() + .01 * (i % 2)
        module.optim.latest_grad_norm = expected_norm.detach() + .01 * (i % 2)
    module.zero_grads()
    module.train_batch({'input_ids': inputs, 'labels': labels})
    module.optim_step()
    assert metrics[('optim', 'step skipped')].item() == 0
    assert any(not torch.equal(parameter, original) for parameter, original in zip(parameters, before))
    assert all(module.optim.state[parameter]['step'].item() == 1 for parameter in parameters)


def test_tied_initialization_matches_emo_final_output_draw():
    # EMO initializes embeddings, then blocks, then the output projection even
    # when it aliases embeddings. The untied twin exposes both seeded draws.
    untied = tiny_model(tied=False)
    current = tiny_model()
    legacy = tiny_model(emo=False)
    for model in (untied, current, legacy):
        model.init_weights()
    assert current.embeddings.weight is current.lm_head.w_out.weight
    assert legacy.embeddings.weight is legacy.lm_head.w_out.weight
    torch.testing.assert_close(current.embeddings.weight, untied.lm_head.w_out.weight, rtol=0, atol=0)
    torch.testing.assert_close(legacy.embeddings.weight, untied.embeddings.weight, rtol=0, atol=0)
    assert not torch.equal(current.embeddings.weight, legacy.embeddings.weight)
    for name, parameter in current.blocks.named_parameters():
        torch.testing.assert_close(parameter, dict(untied.blocks.named_parameters())[name], rtol=0, atol=0)


@pytest.mark.parametrize('tied', [False, True])
def test_fsdp_sharding_keeps_shared_parameter_in_root(monkeypatch, tied):
    import olmo_core.nn.transformer.model as implementation
    model = tiny_model(tied=tied)
    groups, blocks = [], []
    def shard(module, **kwargs):
        groups.append((module, kwargs))
        module.set_unshard_in_backward = lambda _: None
    monkeypatch.setattr(implementation, 'fully_shard', shard)
    for block in model.blocks.values():
        monkeypatch.setattr(block, 'apply_fsdp', lambda **kwargs: blocks.append(kwargs))
    model.apply_fsdp(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    modules = [m for m, _ in groups]
    assert modules[-1] is model
    assert (model.embeddings in modules) is (not tied)
    assert (model.lm_head in modules) is (not tied)
    assert len(blocks) == 1
    assert blocks[0]['wrapping_strategy'] == TransformerDataParallelWrappingStrategy.full
    for _, settings in groups:
        assert settings['mp_policy'].param_dtype == torch.bfloat16
        assert settings['mp_policy'].reduce_dtype == torch.float32
    if tied:
        assert model.embeddings.weight is model.lm_head.w_out.weight
        with pytest.raises(NotImplementedError):
            model.apply_fsdp(pp_enabled=True)
        with pytest.raises(NotImplementedError):
            model.apply_tp(None)


def test_new_fingerprint_and_same_recipe_across_train_convert_eval(tmp_path):
    args = make_meta_fixture(tmp_path, 32000, .5, 4_000_000)
    config, output = pipeline.training_plan(args)
    assert config['training_recipe'] == EMO_SETTINGS
    for stage in pipeline.MODEL_STAGES:
        downstream = copy.deepcopy(args)
        downstream.stage = stage
        if stage == 'evaluate-model':
            downstream.evals = list(pipeline.EVALS)
        serialized = submit.stage_arguments(downstream)
        parsed = submit.parser().parse_args(serialized)
        assert parsed.training_recipe == EMO_RECIPE
        assert pipeline.training_plan(parsed) == (config, output)
    args.training_recipe = LEGACY_RECIPE
    legacy, old_output = pipeline.training_plan(args)
    assert old_output != output
    without_recipe = copy.deepcopy(config)
    del without_recipe['training_recipe'], without_recipe['model']['init_tied_output']
    assert legacy == without_recipe


def test_historical_campaign_commands_and_explicit_new_recipe():
    for stage in pipeline.MODEL_STAGES:
        node = {'stage': stage, 'arguments': [stage, '--vocab-size', '32000']}
        original = copy.deepcopy(node)
        args = submit.parser().parse_args(campaign_arguments(node))
        assert args.training_recipe == LEGACY_RECIPE
        assert node == original
        node['arguments'] += ['--training-recipe', EMO_RECIPE]
        assert campaign_arguments(node) == node['arguments']


def test_old_health_decisions_preserved_new_recipe_skips_and_continues():
    config = small_config()
    config['training']['health_check'] = True
    configure_recipe(config)
    assert config['training']['health_check'] is False
    node = {'arguments': [], 'loss_curve_review': {'status': 'unhealthy'},
            'gradient_curve_review': {'status': 'unhealthy'}}
    assert invalid_result(node)
    node['arguments'] = ['--training-recipe', EMO_RECIPE]
    assert not invalid_result(node)
    node['loss_health_decision'] = {'status': 'unhealthy'}
    assert invalid_result(node)
    node['loss_health_decision'] = {'status': 'valid'}
    assert not invalid_result(node)


def test_checkpoint_recipe_cannot_be_changed_on_resume(tmp_path):
    legacy = small_config()
    current = configure_recipe(copy.deepcopy(legacy))
    (tmp_path / 'config.json').write_text(json.dumps(legacy))
    checkpoint_recipe_guard(tmp_path, legacy)
    with pytest.raises(ValueError, match='recipe mismatch'):
        checkpoint_recipe_guard(tmp_path, current)
    (tmp_path / 'config.json').write_text(json.dumps(current))
    checkpoint_recipe_guard(tmp_path, current)
    with pytest.raises(ValueError, match='recipe mismatch'):
        checkpoint_recipe_guard(tmp_path, legacy)


@pytest.mark.parametrize('field,value', [('betas', [.9, .999]), ('eps', 1e-6),
    ('z_loss_multiplier', 0), ('autocast_precision', 'bfloat16'), ('float32_matmul_precision', 'high')])
def test_conflicting_json_cannot_be_silently_ignored(field, value):
    config = small_config()
    config['training'][field] = value
    with pytest.raises(ValueError, match='conflicts'):
        configure_recipe(config)


def test_live_verification_catches_unwrapped_model_and_restored_betas():
    model = tiny_model(emo=False)
    optimizer = optimizer_config(small_config()['training']).build(model)
    module = SimpleNamespace(model=model, optim=optimizer)
    with pytest.raises(RuntimeError, match='actual FSDP'):
        verify_live_recipe(module, small_config())
    module.optim = optimizer_config(small_config()['training'], recipe=LEGACY_RECIPE, fused=False).build(model)
    verify_live_recipe(module, small_config(), recipe=LEGACY_RECIPE)
    module.optim.param_groups[0]['betas'] = (.9, .95)
    with pytest.raises(RuntimeError, match='Restored optimizer'):
        verify_live_recipe(module, small_config(), recipe=LEGACY_RECIPE)


def test_main_passes_emo_settings_to_actual_build_boundary(tmp_path, monkeypatch):
    import train
    args = make_meta_fixture(tmp_path, 32000, .5, 4_000_000)
    config, output = pipeline.training_plan(args)
    meta_path = pipeline.tokenized_dir(args) / 'meta.json'
    meta = pipeline.read_json(meta_path)
    meta.update(eos_token_id=0, eos_token='<eos>')
    pipeline.write_json(meta_path, meta)
    config_path = tmp_path / 'launch.json'
    pipeline.write_json(config_path, config)
    monkeypatch.setenv('WORLD_SIZE', '8')
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(config_path),
        '--data-dir', str(pipeline.tokenized_dir(args)), '--output-dir', str(output)])
    monkeypatch.setattr(train, 'prepare_training_environment', lambda **_: None)
    monkeypatch.setattr(train, 'teardown_training_environment', lambda: None)
    captured = {}
    class BuildBoundary(Exception):
        pass
    def model_build(self, **kwargs):
        captured['model_config'] = self
        return SimpleNamespace()
    def module_build(self, model):
        captured['module_config'] = self
        captured['precision'] = torch.get_float32_matmul_precision()
        raise BuildBoundary()  # Stop before allocating a model or running training.
    monkeypatch.setattr(train.TransformerConfig, 'build', model_build)
    monkeypatch.setattr(train.TransformerTrainModuleConfig, 'build', module_build)
    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        with pytest.raises(BuildBoundary):
            train.main()
    finally:
        torch.set_float32_matmul_precision(original)
    assert captured['precision'] == 'highest'
    assert captured['model_config'].init_tied_output
    module = captured['module_config']
    assert isinstance(module.optim, SkipStepAdamWConfig) and module.optim.betas == (.9, .95)
    assert module.dp_config.name == DataParallelType.fsdp and module.autocast_precision is None
    assert module.z_loss_multiplier == 1e-5


def test_callback_records_live_objects_and_rejects_restore_drift(tmp_path):
    config = small_config()
    model_cfg = TransformerConfig.llama_like(d_model=16, n_layers=1, n_heads=2, vocab_size=128,
        tie_word_embeddings=True, attn_backend=AttentionBackendName.torch)
    model = model_cfg.build(init_device='cpu')
    module_cfg = train_module_config(config, 1, recipe=LEGACY_RECIPE)
    optimizer = optimizer_config(config['training'], recipe=LEGACY_RECIPE, fused=False).build(model)
    module = SimpleNamespace(model=model, optim=optimizer, z_loss_multiplier=None,
                             autocast_precision=torch.bfloat16)
    callback = TrainingRecipeCallback(config, model_cfg, module_cfg, tmp_path, LEGACY_RECIPE)
    callback.trainer = SimpleNamespace(train_module=module, callbacks={'training_recipe': callback})
    callback.pre_train()
    record = json.loads((tmp_path / 'resolved_training_config.json').read_text())
    assert record['actual_optimizer'] == 'AdamW' and record['actual_betas'] == [[.9, .999]]
    optimizer.param_groups[0]['eps'] = 1e-4
    with pytest.raises(RuntimeError, match='Restored optimizer'):
        callback.pre_train()


def test_emo_checkpoint_converts_with_tied_weights_and_matching_logits(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from olmo_core.distributed.checkpoint import save_model_and_optim_state
    from eval.checkpoint_converter import convert
    config = small_config()
    config['model'].update(d_model=16, n_layers=1, n_heads=2)
    config['seq_length'] = 16
    config['eos_token'] = '<eos>'
    configure_recipe(config)
    checkpoint = tmp_path / 'run' / 'step1'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text(json.dumps(config))
    tokenizer = Tokenizer(models.WordLevel({'<eos>': 0, '<unk>': 1, 'test': 2}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(checkpoint.parent / 'tokenizer.json'))
    model = TransformerConfig.llama_like(vocab_size=128, attn_backend=AttentionBackendName.torch,
                                        **config['model']).build(init_device='cpu')
    model.init_weights()
    save_model_and_optim_state(checkpoint / 'model_and_optim', model)
    convert(checkpoint, tmp_path / 'converted')
    destination = tmp_path / 'converted/run/step1'
    evidence = json.loads((destination / 'conversion.json').read_text())
    assert evidence['max_logit_error'] < 1e-4
    assert json.loads((destination / 'config.json').read_text())['tie_word_embeddings'] is True
