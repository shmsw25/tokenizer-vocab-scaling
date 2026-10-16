import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gradient_health as health
from loss_curve_health import invalid_result
from test_report import state_fixture
import report


def curve(peak=3., step=400):
    return [{'_step': i, health.POLICY['metric']: peak if i == step else .25}
            for i in range(1, 601)]


def test_user_thresholds_single_spikes_survive_recovery_and_do_not_need_ce():
    severe = health.analyze_history(curve(2.5))
    assert severe['status'] == 'unhealthy'
    assert severe['spikes'][0]['multiple'] == 10
    assert severe['last_step'] == 600
    assert health.analyze_history(curve(2.499))['status'] == 'review_needed'
    assert health.analyze_history(curve(1.25))['status'] == 'review_needed'
    assert health.analyze_history(curve(1.249))['status'] == 'no_spike_detected'
    assert invalid_result({'gradient_curve_review': severe})
    assert not invalid_result({'gradient_curve_review': severe, 'loss_health_decision': {'status': 'valid'}})


def test_warmup_missing_nonfinite_duplicate_steps_and_zero_baseline():
    assert health.analyze_history(curve(20., 291))['status'] == 'no_spike_detected'
    assert health.analyze_history(curve(20., 292))['status'] == 'unhealthy'
    rows = curve(.25) + [{'_step': 601, health.POLICY['metric']: None}]
    result = health.analyze_history(rows)
    assert result['missing_values'] == 1 and result['status'] == 'no_spike_detected'
    assert health.analyze_history(curve(float('nan')))['status'] == 'review_needed'
    assert health.analyze_history([])['status'] == 'unavailable'
    assert health.analyze_history(curve() + [curve()[399]])['observations'] == 600
    with pytest.raises(ValueError, match='Conflicting'):
        health.analyze_history(curve() + [{'_step': 400, health.POLICY['metric']: 9.}])
    zero = [{'_step': i, health.POLICY['metric']: 0. if i != 400 else 1.} for i in range(1, 601)]
    assert health.analyze_history(zero)['status'] == 'review_needed'
    # A single measured 10x spike is enough, even with sparse logging.
    assert health.analyze_history([r for r in curve() if r['_step'] % 2 == 0])['status'] == 'unhealthy'


def test_prior_attempts_independent_metric_query_cache_integrity_and_archive(tmp_path):
    calls = []
    class Run:
        def __init__(self, path): self.path = path; self.url = 'https://example.test/recorded/' + path
        def scan_history(self, **kwargs):
            assert kwargs['keys'] == ['_step', '_timestamp', health.POLICY['metric']]
            calls.append(self.path)
            return curve(3. if self.path.endswith('old') else .25)
    api = SimpleNamespace(run=Run)
    node = {'status': 'succeeded', 'reported_metrics': {'wandb_path': 'e/p/current'},
            'previous_attempts': [{'reported_metrics': {'wandb_path': 'e/p/old'}}]}
    before = copy.deepcopy(node)
    health.review_node(node, api, tmp_path, 'now')
    assert node['gradient_curve_review']['status'] == 'unhealthy'
    assert {k: node[k] for k in before} == before
    assert not health.needs_review(node)
    health.review_node(node, None, tmp_path, 'later')
    assert calls == ['e/p/old', 'e/p/current']
    assert node['gradient_curve_review']['sources'][0]['url'] == 'https://example.test/recorded/e/p/old'
    cache = Path(node['gradient_curve_review']['sources'][1]['history_file'])
    prior_bytes = cache.read_bytes()
    node['status'] = 'running'
    health.review_node(node, api, tmp_path, 'live')
    assert (tmp_path / 'history' / (hashlib.sha256(prior_bytes).hexdigest() + '.json')).read_bytes() == prior_bytes
    node['status'] = 'succeeded'
    health.review_node(node, api, tmp_path, 'final')
    payload = json.loads(cache.read_text()); payload['rows'][0][health.POLICY['metric']] = 10
    cache.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='hash mismatch'):
        health.review_node(node, None, tmp_path, 'changed')


def test_report_blanks_gradient_unhealthy_and_keeps_review_scores():
    s = state_fixture(); key = 'train-small-5gb-32000'; node = s['nodes'][key]
    node.update(status='succeeded', gradient_curve_review=health.analyze_history(curve()),
                reported_metrics={'validation_bpb': .8})
    data = report.build_report_data(s)
    row = next(r for r in data['runs'] if r['id'] == key)
    assert row['status'] == 'unhealthy' and key not in data['selected']
    assert all(value is None for value in row['metrics'].values())
    assert row['gradient_curve_review']['spikes']
    node['gradient_curve_review'] = health.analyze_history(curve(1.5))
    row = next(r for r in report.build_report_data(s)['runs'] if r['id'] == key)
    assert row['status'] == 'review' and row['result_valid']


def test_native_guard_stops_single_10x_even_without_ce_and_final_step(tmp_path):
    from scripts.training_callbacks import LossHealthCallback
    for peak in (1.5, 2.5):
        cb = LossHealthCallback(str(tmp_path / str(peak)), warmup_steps=192)
        canceled = []; cb.trainer = SimpleNamespace(cancel_run=canceled.append)
        for r in curve(peak, 600):
            # CE can arrive first, followed by the norm at the same optimizer step.
            cb.log_metrics(r['_step'], {'train/CE loss': 3.})
            cb.log_metrics(r['_step'], {health.POLICY['metric']: r[health.POLICY['metric']]})
        assert bool(canceled) == (peak == 2.5)
        if canceled:
            stop = json.loads((Path(cb.output_dir) / 'health_stop.json').read_text())
            assert stop['step'] == 600 and stop['gradient_curve_evidence']['multiple'] == 10
            assert invalid_result({'health_stop': stop})
            cb.log_metrics(601, {health.POLICY['metric']: 25.})
            assert len(canceled) == 1


def test_native_guard_nulls_nonfinite_and_resume_grace(tmp_path):
    from scripts.training_callbacks import LossHealthCallback
    cb = LossHealthCallback(str(tmp_path), warmup_steps=192)
    calls = []; cb.trainer = SimpleNamespace(cancel_run=calls.append, global_step=500)
    cb.log_metrics(1, {health.POLICY['metric']: None})
    assert not calls
    cb.gradient_state = {'last_step': 500, 'values': [.25] * 100}
    cb.pre_train()
    cb.log_metrics(501, {health.POLICY['metric']: 20.})
    assert not calls
    cb.log_metrics(502, {health.POLICY['metric']: float('nan')})
    assert len(calls) == 1


def test_gradient_verdict_triggers_only_authorized_terminal_small_10gb_fallback():
    from test_adaptive_study import descent_fixture
    import adaptive_study as study
    s, gate_key = descent_fixture()
    s['nodes'][gate_key].update(status='succeeded', chain_job_status='succeeded')
    required = set(); study.advance_small_10gb_lr_descent(s, required)
    runs = s['study']['small_10gb_lr_descent']['runs']['10GB:small:32000']
    n = s['nodes'][runs[0]]
    n.update(status='running', chain_job_status='running', receipt={'workload_id':'original'},
             gradient_curve_review=health.analyze_history(curve()))
    study.advance_small_10gb_lr_descent(s, required)
    assert len(runs) == 1
    n.update(status='canceled', chain_job_status='canceled')
    study.advance_small_10gb_lr_descent(s, required)
    assert len(runs) == 2 and float(study.arg(s['nodes'][runs[-1]], '--learning-rate')) == .0002
    assert len(s['study']['small_10gb_lr_descent']['runs']['10GB:small:64000']) == 1


def test_training_hold_blocks_future_training_but_preserves_cpu_and_validation(monkeypatch, tmp_path):
    import campaign
    stages = ('train-model', 'validate-model', 'convert-model', 'tokenize-data')
    s = {'study': {'training_submission_hold': {'reason': 'User asked to wait'}},
         'nodes': {stage: {'stage': stage, 'status': 'pending', 'dependencies': [], 'arguments': []}
                   for stage in stages}}
    assert set(campaign.ready_nodes(s)) == set(stages) - {'train-model'}
    # Admission checks the hold again even if a stale candidate list includes training.
    monkeypatch.setattr(campaign, 'ready_nodes', lambda state: ['train-model'])
    import adaptive_study
    def no_budget_query(state): raise AssertionError('Held training reached admission')
    monkeypatch.setattr(adaptive_study, 'live_gpu_budget', no_budget_query)
    before = copy.deepcopy(s)
    campaign.submit_ready(s, tmp_path / 'state.json')
    assert s == before


def test_confirmed_manually_deleted_held_jobs_keep_receipts_without_recreation(monkeypatch, tmp_path):
    from beaker import Beaker
    from unittest.mock import MagicMock
    import campaign
    node = {'stage': 'train-model', 'status': 'canceled', 'user_hold': True,
            'receipt': {'workload_id': 'deleted'}, 'user_hold_cancellation': {
                'workload_id': 'deleted', 'workload_lookup': 'not_found', 'confirmed_at': 'now',
                'job_status': {'status': 'STATUS_CANCELED'}}}
    state = {'workspace': 'workspace', 'owner': 'owner', 'nodes': {'training': node}}
    before = copy.deepcopy(node)
    client = MagicMock(); client.__enter__.return_value = client
    monkeypatch.setattr(Beaker, 'from_env', lambda **kwargs: client)
    campaign.refresh(state, tmp_path)
    client.workload.get.assert_not_called()
    assert node == before
    node['user_hold_cancellation']['workload_id'] = 'wrong'
    client.workload.get.side_effect = ValueError('Unconfirmed receipt')
    campaign.refresh(state, tmp_path)
    client.workload.get.assert_called_once_with('deleted')
    assert node['monitor_error']
