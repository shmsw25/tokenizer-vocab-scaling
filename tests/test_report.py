import copy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import campaign
import pipeline
import report
from training_callbacks import ReportMetricsCallback


def state_fixture():
    return {'complete': False, 'pool_sizes': list(pipeline.POOL_SIZES),
            'training_pools': list(pipeline.TRAINING_POOLS),
            'nodes': campaign.make_nodes([32000, 64000, 128000], report.MODEL_ORDER)}


def metadata(tokens=10_000_000, ratio=.25):
    return {'data_files': ['same-pool.jsonl'], 'seed': 42, 'eval_fraction': .01,
            'tokens_per_byte': ratio,
            'train': {'tokens': tokens, 'bytes': 40_000_000, 'documents': 1000},
            'eval': {'tokens': 100_000, 'bytes': 400_000, 'documents': 10}}


def test_report_selects_only_authorized_lower_pool_trials_and_honors_user_verdict():
    import adaptive_study as study
    state = state_fixture()
    study.initialize(state)
    state['study']['pool_lr_search'] = {'pools': ['10GB', '20GB'], 'learning_rates': [.001]}
    old = study.model_run(state, '10GB', 'small', 128000, .001, role='existing')
    lower = study.model_run(state, '10GB', 'small', 128000, .0005, role='small_10gb_lr_descent')
    unrelated = study.model_run(state, '10GB', 'small', 128000, .0002, role='unapproved')
    state['study']['small_10gb_lr_descent'] = {'runs': {'10GB:small:128000': [lower]}}
    for key, score in [(old, .7), (lower, .8), (unrelated, .6)]:
        state['nodes'][key].update(status='succeeded', reported_metrics={'validation_bpb': score})
    state['nodes'][old]['loss_health_decision'] = {'status': 'unhealthy', 'source': 'user'}
    data = report.build_report_data(state)
    assert lower in data['selected'] and old not in data['selected'] and unrelated not in data['selected']
    invalid = next(row for row in data['runs'] if row['id'] == old)
    assert invalid['status'] == 'unhealthy'
    assert all(value is None for value in invalid['metrics'].values())


def test_exact_emo_compute_and_epoch_tail():
    node = state_fixture()['nodes']['train-small-5gb-32000']
    cfg = report.run_config(node)
    meta = metadata()
    assert report.compute_budget(node, cfg, meta, {})['epoch_flops'] is None
    budget = report.compute_budget(node, cfg, meta, meta)
    # Attention projections, SwiGLU, output projection/norm, attention products.
    d, layers, hidden, vocab, seq = 512, 12, 2048, 32000, 4096
    expected = 6 * (layers * (4*d*d + 3*d*hidden) + d*vocab + d) + 12*layers*d*seq
    assert budget['flops_per_token'] == expected
    assert budget['epoch_tokens'] == 19 * 4096 * 128
    assert budget['epoch_flops'] == budget['epoch_tokens'] * expected
    assert budget['dropped_tokens'] == 10_000_000 - budget['epoch_tokens']
    target = metadata(8_000_000, .20)
    larger = dict(node, vocab_size=128000)
    scaled = report.compute_budget(larger, cfg, target, meta)
    assert scaled['seq_length'] == round(4096 * .8)
    assert scaled['flops_per_token'] > expected
    wrong = copy.deepcopy(meta)
    wrong['train']['documents'] += 1
    with pytest.raises(ValueError, match='train.documents'):
        report.compute_budget(node, cfg, meta, wrong)


def test_counts_variants_selection_and_plot_coordinates(tmp_path):
    state = state_fixture()
    nodes = state['nodes']
    for vocab, status in [(32000, 'succeeded'), (64000, 'initializing'), (128000, 'submitted')]:
        nodes[f'bpe-5gb-{vocab}']['status'] = status
    meta = metadata()
    nodes['tokens-5gb-32000'].update(status='succeeded', summary={'meta': meta})
    for i, model in enumerate(report.MODEL_ORDER):
        nodes[f'train-{model}-5gb-32000'].update(status='succeeded', reported_metrics={'validation_bpb': 2-i*.1})
        nodes[f'eval-{model}-5gb-32000']['summary'] = {'metrics': {'dolma_bpb': {'bpb': 1.5-i*.1}}}
    variant = copy.deepcopy(nodes['train-small-5gb-32000'])
    variant['arguments'][variant['arguments'].index('--learning-rate') + 1] = '.001'
    variant['reported_metrics'] = {'validation_bpb': 3.0}
    nodes['train-small-5gb-32000-lr001'] = variant
    data = report.build_report_data(state)
    assert len(data['runs']) == 28
    assert len(set((r['pool'], r['vocab'], r['model']) for r in data['runs'])) == 27
    count = next(c for c in data['tokenizers'] if c['pool'] == '5GB')
    assert (count['completed'], count['running'], count['queued']) == (1, 1, 1)
    assert 'train-small-5gb-32000' in data['selected']
    assert 'train-small-5gb-32000-lr001' not in data['selected']
    assert {r['vocab_label'] for r in data['runs']} == {'32K', '64K', '128K'}
    figs = report.build_figures(data)
    assert len(figs['dolma_bpb']) == 3
    trace = figs['dolma_bpb'][0]['data'][0]
    assert trace['y'] == [1.5, 1.4, 1.3]
    assert list(trace['x']) == [next(r for r in data['runs'] if r['id'] == f'train-{m}-5gb-32000')['compute']['epoch_flops'] for m in report.MODEL_ORDER]
    state['notes'] = '</script><script>alert(1)</script>'
    path = tmp_path / 'report.html'
    report.render_report(state, path)
    html = path.read_text()
    assert '<h2>Evaluation results</h2>' not in html
    assert "$('metric').value='dolma_bpb'" in html
    assert 'Details +' not in html and '<details class="coordinate' not in html
    assert state['notes'] not in html


def test_loss_metrics_keep_latest_observation_and_resume(tmp_path):
    cb = ReportMetricsCallback(str(tmp_path), .25, .2)
    cb.log_metrics(20, {'train/CE loss': 3., 'eval/lm/validation/CE loss': 4.})
    cb.log_metrics(10, {'train/CE loss': 9.})
    cb.log_metrics(30, {'train/CE loss': float('nan'), 'other': 5.})
    saved = json.loads((tmp_path / 'report_metrics.json').read_text())
    assert saved['train_step'] == 20
    assert saved['validation_bpb'] == pytest.approx(.8 / math.log(2))
    resumed = ReportMetricsCallback(str(tmp_path), .25, .2)
    resumed.pre_train()
    assert resumed.latest == saved


def test_legacy_medium_large_results_are_excluded():
    state=state_fixture()
    for node in state['nodes'].values():
        if node.get('model_size')=='medium_large':node['legacy_architecture']=True
    data=report.build_report_data(state)
    assert not any(r['model']=='medium_large' for r in data['runs'])
    assert any(r['model']=='medium' for r in data['runs'])


def test_dclm_counts_use_matching_full_pools_and_training_split_only():
    import dclm_campaign
    dclm = dclm_campaign.make_state()
    dclm_campaign.enable_tokenization(dclm)
    raw = {'records': 1010, 'text_bytes': 40_400_000}
    dclm['nodes']['dclm-pools']['summary'] = {'pools': {'5GB': raw}}
    for vocab, count in [(32000, 10_000_000), (64000, 9_000_000)]:
        meta = metadata(count)
        meta.update(vocab_size=vocab, eos_token_id=0)
        info = {'meta': meta, 'recipe': {'pool': raw, 'tokenizer_sha256': str(vocab)},
                'total_tokens': count+100_000,
                'verified': dict.fromkeys(('full_pool', 'token_stream_hashes', 'document_metadata_hashes', 'tokenizer_hash'), True)}
        dclm['nodes'][f'dclm-bpe-5gb-{vocab}']['summary'] = {'tokenizer': {'output': {'sha256': str(vocab)}}}
        dclm['nodes'][f'dclm-tokens-5gb-{vocab}'].update(status='succeeded', summary={'meta': meta, 'tokenization': info})
    model = state_fixture()
    dolma_before = report.build_report_data(model)
    result = report.build_report_data(model, dclm)
    row = next(r for r in result['training_tokens'] if r['dataset'] == 'DCLM' and r['pool'] == '5GB')
    assert row['tokens'] == {'32K': 10_000_000, '64K': 9_000_000, '128K': None}
    assert row['change_vs_32k_pct']['64K'] == pytest.approx(-10)
    assert row['all_tokens']['32K'] == 10_100_000
    assert row['change_vs_32k_pct']['128K'] is None
    assert result['runs'] == dolma_before['runs']
    assert [r for r in result['training_tokens'] if r['dataset'] == 'Dolma'] == dolma_before['training_tokens']
    changed = dclm['nodes']['dclm-tokens-5gb-64000']['summary']['meta']
    changed['train']['documents'] -= 1
    changed['eval']['documents'] += 1
    with pytest.raises(ValueError, match='documents differ'):
        report.build_report_data(model, dclm)
