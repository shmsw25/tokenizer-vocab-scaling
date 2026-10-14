import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import adaptive_study
import campaign
import pipeline
import report
import submit
from train import optimizer_config, verify_optimizer_groups
from test_pipeline import make_meta_fixture
from test_report import state_fixture


@pytest.mark.parametrize('embedding_wd', [None, 0.0, 0.03, 0.1])
def test_actual_tied_embedding_update_and_other_parameter_decay(embedding_wd):
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig
    model = TransformerConfig.llama_like(d_model=32, n_layers=1, n_heads=4,
        vocab_size=256, tie_word_embeddings=True,
        attn_backend=AttentionBackendName.torch).build(init_device='cpu')
    model.init_weights()
    config = {'learning_rate': .01, 'weight_decay': .1}
    if embedding_wd is not None:
        config['embedding_weight_decay'] = embedding_wd
    optimizer = optimizer_config(config, fused=False).build(model)
    evidence = verify_optimizer_groups(model, optimizer, config)
    assert model.embeddings.weight is model.lm_head.w_out.weight
    expected = .1 if embedding_wd is None else embedding_wd
    assert len(evidence) == (1 if expected == .1 else 2)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    for name, p in model.named_parameters():
        decay = expected if p is model.embeddings.weight else .1
        torch.testing.assert_close(p, before[name] * (1 - .01 * decay))
    assert sum(g['includes_embedding'] for g in evidence) == 1
    optimizer.param_groups[0]['weight_decay'] = .7
    with pytest.raises(RuntimeError, match='recipe'):
        verify_optimizer_groups(model, optimizer, config)


def test_fingerprints_and_all_chain_stages_preserve_override(tmp_path):
    args = make_meta_fixture(tmp_path, 32000, .5, 4_000_000)
    original, original_path = pipeline.training_plan(args)
    args.embedding_weight_decay = .1
    assert pipeline.training_plan(args) == (original, original_path)
    args.embedding_weight_decay = 0
    config, output = pipeline.training_plan(args)
    assert output != original_path and '_EmbedWD0_' in output.name
    expected = copy.deepcopy(original)
    expected['training']['embedding_weight_decay'] = 0
    assert config == expected
    for stage in pipeline.MODEL_STAGES:
        downstream = copy.deepcopy(args)
        downstream.stage = stage
        if stage == 'evaluate-model':
            downstream.evals = list(pipeline.EVALS)
        assert pipeline.training_plan(downstream) == (config, output)
        serialized = submit.stage_arguments(downstream)
        assert serialized[serialized.index('--embedding-weight-decay') + 1] == '0'
    for invalid in (-.1, float('nan'), float('inf')):
        args.embedding_weight_decay = invalid
        with pytest.raises(ValueError, match='embedding-weight-decay'):
            pipeline.validate_args(args)


def test_variant_is_distinct_reused_and_compared_within_same_coordinate():
    state = state_fixture()
    state['study'] = {'root': pipeline.DEFAULT_ROOT, 'fixed_learning_rates': {'medium_large': .001},
                     'selected': {'small:32000': {'lr': .002}}, 'required': []}
    baseline = 'train-small-5gb-32000'
    state['nodes'][baseline].update(status='succeeded', receipt={'workload_id': 'original'},
        reported_metrics={'validation_bpb': 1.1})
    key = adaptive_study.model_run(state, '5GB', 'small', 32000, .002,
        role='hyperparameter_trial', weight_decay=.1, batch_size=128, embedding_weight_decay=0)
    assert key != baseline
    assert adaptive_study.model_run(state, '5GB', 'small', 32000, .002,
        role='lr_baseline') == baseline
    assert adaptive_study.model_run(state, '5GB', 'small', 32000, .002,
        role='hyperparameter_trial', embedding_weight_decay=0) == key
    for prefix in ('train', 'convert', 'eval'):
        n = state['nodes'][key.replace('train-', prefix+'-', 1)]
        args = submit.parser().parse_args(n['arguments'])
        pipeline.validate_args(args)
        assert args.weight_decay == .1 and args.embedding_weight_decay == 0
        assert args.batch_size == 128 and args.learning_rate == .002
    campaign.prepare_chains(state)
    assert state['nodes'][adaptive_study.eval_key(key)]['chain_parent'] == key
    state['nodes'][key].update(status='succeeded', reported_metrics={'validation_bpb': 1.0})
    data = report.build_report_data(state)
    runs = [r for r in data['runs'] if r['pool']=='5GB' and r['model']=='small' and r['vocab']==32000]
    assert {r['embedding_wd'] for r in runs} == {0, .1}
    assert key in data['selected'] and baseline not in data['selected']
    assert len({(r['pool'], r['model'], r['vocab']) for r in data['runs']}) == 27
