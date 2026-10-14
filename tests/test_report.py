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
