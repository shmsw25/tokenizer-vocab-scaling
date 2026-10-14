import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import adaptive_study as study
import campaign
import report
from loss_curve_health import analyze_history, invalid_result, needs_review, review_node
from loss_curve_health import beaker_history_rows, cache_beaker_history, discover_beaker_history, logged_wandb_links
from test_adaptive_study import fixture, finish
from test_report import state_fixture


def curve(spike=10., count=3):
    return [{'_step': i, 'train/CE loss': spike if 400 <= i < 400 + count else 3.}
            for i in range(1, 601)]


def test_clear_explosion_stays_unhealthy_after_recovery_and_gray_area_is_review_only():
    clear = analyze_history(curve())
    assert clear['status'] == 'unhealthy' and clear['longest_severe_streak'] == 3
    assert clear['last_step'] == 600 and clear['episodes'][0]['peak']['step'] == 400
    gray = analyze_history(curve(5.43, 2))
    assert gray['status'] == 'review_needed'
    assert not invalid_result({'loss_curve_review': gray})
    assert invalid_result({'loss_curve_review': clear})
    assert not invalid_result({'loss_curve_review': clear, 'loss_health_decision': {'status': 'valid'}})


def test_sustained_smaller_spikes_like_cs48vsyq_are_unhealthy_but_brief_spikes_need_review():
    # The user example rose from ~3.13 to 5.43, with 14 adjacent steps above
    # 1.5x baseline. A recovered endpoint must not erase this smaller explosion.
    rows = [{'_step': i, 'train/CE loss': 5.43 if 400 <= i < 414 else 3.13}
            for i in range(1, 601)]
    result = analyze_history(rows)
    assert result['status'] == 'unhealthy'
    assert result['longest_spike_streak'] == 14
    assert result['longest_severe_streak'] == 0
    assert analyze_history(curve(5.43, 5))['status'] == 'unhealthy'
    assert analyze_history(curve(5.43, 4))['status'] == 'review_needed'
    # Large relative increases alone at low CE do not satisfy the absolute floor.
    low = [{'_step': i, 'train/CE loss': 1.6 if 400 <= i < 420 else 1.}
           for i in range(1, 601)]
    assert analyze_history(low)['status'] == 'no_spike_detected'
    # Missing steps and explicit nonfinite values break persistence.
    interrupted = [r for r in curve(5.43, 9) if r['_step'] != 404]
    assert analyze_history(interrupted)['status'] == 'review_needed'
    interrupted.append({'_step': 404, 'train/CE loss': float('nan')})
    assert analyze_history(interrupted)['status'] == 'review_needed'


def test_normal_noise_warmup_nulls_and_sparse_history_do_not_auto_invalidate():
    rows = [{'_step': i, 'train/CE loss': 12. if i < 150 else 3. + .1 * (i % 3 - 1)}
            for i in range(1, 601)]
    rows.append({'_step': 601, 'train/CE loss': None})
    result = analyze_history(rows)
    assert result['status'] == 'no_spike_detected' and result['missing_values'] == 1
    sparse = [r for r in curve(10., 6) if r['_step'] % 2 == 0]
    assert analyze_history(sparse)['status'] == 'review_needed'
    assert analyze_history([])['status'] == 'unavailable'


def test_complete_history_and_pre_resume_attempt_are_audited_and_cached(tmp_path):
    calls = []
    class Run:
        def __init__(self, path): self.path = path
        def scan_history(self, **kwargs):
            calls.append(self.path)
            assert kwargs['keys'] == ['_step', '_timestamp', 'train/CE loss']
            return iter(curve() if self.path.endswith('old') else curve(3.))
    api = SimpleNamespace(run=lambda path: Run(path))
    node = {'status': 'succeeded', 'reported_metrics': {'wandb_path': 'entity/project/current'},
            'previous_attempts': [{'reported_metrics': {'wandb_path': 'entity/project/old'}}],
            'loss_health_decision': {'status': 'valid', 'reason': 'human override'}}
    original = copy.deepcopy(node)
    review_node(node, api, tmp_path, 'now')
    assert node['loss_curve_review']['status'] == 'unhealthy'
    assert node['loss_curve_review']['sources'][0]['url'] == 'https://wandb.ai/entity/project/runs/old'
    assert not invalid_result(node) and not needs_review(node)
    review_node(node, api, tmp_path, 'later')
    assert calls == ['entity/project/old', 'entity/project/current']
    assert {k:node[k] for k in original} == original
    node['status'] = 'running'
    node['loss_curve_review']['terminal'] = False
    assert needs_review(node)


def test_invalid_scores_are_blank_and_excluded_but_review_scores_and_provenance_remain():
    s = state_fixture(); key = 'train-small-5gb-32000'; n = s['nodes'][key]
    n.update(status='succeeded', reported_metrics={'validation_bpb': .8},
             loss_curve_review={'status': 'unhealthy'})
    s['nodes'][study.eval_key(key)]['summary'] = {'metrics': {'dolma_bpb': {'bpb': .9}}}
    data = report.build_report_data(s)
    run = next(r for r in data['runs'] if r['id'] == key)
    assert run['status'] == 'unhealthy' and run['training_status'] == 'completed'
    assert all(v is None for v in run['metrics'].values())
    assert run['raw_metrics']['dolma_bpb'] == .9 and key not in data['selected']
    assert n['reported_metrics']['validation_bpb'] == .8
    n['loss_curve_review']['status'] = 'review_needed'
    data = report.build_report_data(s); run = next(r for r in data['runs'] if r['id'] == key)
    assert run['status'] == 'review' and run['metrics']['dolma_bpb'] == .9 and key in data['selected']


def test_invalidation_revokes_cached_fixed_lr_winner_without_retraining_or_changing_receipts():
    s = fixture()
    s['study']['fixed_learning_rates'] = {'medium_large': .001}
    s['study']['pool_lr_search'] = {'pools': ['10GB', '20GB'], 'learning_rates': [.001, .002]}
    study.advance_plan(s)
    coord = '10GB:medium_large:32000'; key = s['study']['pool_lr_search']['runs'][coord][0]
    finish(s, key, .9); n = s['nodes'][key]
    n['receipt'] = {'workload_id': 'existing'}
    s['nodes'][study.eval_key(key)]['status'] = 'succeeded'
    study.advance_plan(s)
    assert s['study']['pool_lr_search']['selected'][coord]['key'] == key
    before = set(s['nodes']); n['loss_curve_review'] = {'status': 'unhealthy'}
    study.advance_plan(s)
    assert study.completed_score(n) == float('inf')
    assert coord not in s['study']['pool_lr_search']['selected'] and coord not in s['study']['transfers']
    assert s['study']['phase'] != 'complete' and set(s['nodes']) == before
    assert n['status'] == 'succeeded' and n['receipt']['workload_id'] == 'existing'
    assert not campaign.dependency_ready(s, {}, key)
    assert study.eval_key(key) not in s['study']['required']


def test_pending_infrastructure_investigation_is_still_not_negative_lr_evidence():
    node = {'status': 'failed', 'failure_investigation': {'status': 'unresolved'},
            'health_stop': {'reason': 'Native NaN'}, 'loss_curve_review': {'status': 'review_needed'}}
    assert study.completed_score(node) is None
    node['loss_curve_review']['status'] = 'unhealthy'
    assert study.completed_score(node) is None


def test_user_invalidated_endpoint_is_excluded_from_backfill_without_canceling_validation():
    s = state_fixture(); key = 'train-small-5gb-32000'; n = s['nodes'][key]
    n.update(status='succeeded', loss_curve_review={'status': 'review_needed'},
             loss_health_decision={'status': 'unhealthy', 'source': 'user'})
    validation = key.replace('train-', 'validation-', 1)
    s['nodes'][validation] = {'stage': 'validate-model', 'status': 'queued',
                             'receipt': {'workload_id': 'existing-validation'}}
    s['study'] = {'validation_node': 'prepare-validation',
                  'validation_backfill': {'targets': [key], 'status': 'pending'}}
    before = copy.deepcopy(s['nodes'])
    study.sync_shared_validation(s)
    backfill = s['study']['validation_backfill']
    assert backfill['skipped_unhealthy'] == [key] and backfill['status'] == 'complete'
    assert s['study']['shared_validation_nodes'] == []
    assert set(s['nodes']) == set(before)
    assert s['nodes'][key] == before[key]
    assert s['nodes'][validation] == before[validation]  # Keep the queued receipt intact.


def test_beaker_per_step_history_preserves_adjacency_and_validation_echoes(tmp_path):
    lines = [{'message': 'REPORT_METRICS ' + json.dumps({'train_step': r['_step'],
             'train_ce': r['train/CE loss'], 'step': r['_step']})} for r in curve(7., 8)]
    lines.extend([{'message': 'REPORT_METRICS {"step": 999, "validation_step": 999, "train_step": 600, "train_ce": 3}'},
                  {'message': 'REPORT_METRICS []'}, {'message': 'REPORT_METRICS {"train_step":'}])
    rows = beaker_history_rows(lines)
    assert len(rows) == 600 and rows[-1]['_step'] == 600
    evidence = cache_beaker_history(lines, tmp_path, 'workload', 'job', 'now')
    node = {'status': 'succeeded', 'beaker_loss_histories': [evidence],
            'reported_metrics': {'train_ce': 3.}, 'receipt': {'workload_id': 'workload'}}
    before = copy.deepcopy(node)
    review_node(node, None, tmp_path, 'now')
    assert invalid_result(node) and not needs_review(node)
    assert node['loss_curve_review']['sources'][0]['longest_spike_streak'] == 8
    assert {k:node[k] for k in before} == before
    Path(evidence['history_file']).write_text('{}')
    import pytest
    with pytest.raises(ValueError, match='hash mismatch'):
        review_node(node, None, tmp_path, 'later')


def test_beaker_sparse_or_conflicting_data_never_manufactures_a_sustained_spike():
    import pytest
    lines = [{'message': 'REPORT_METRICS ' + json.dumps({'train_step': r['_step'],
             'train_ce': r['train/CE loss']})} for r in curve(10., 10) if r['_step'] % 2 == 0]
    assert analyze_history(beaker_history_rows(lines))['status'] == 'review_needed'
    lines.append({'message': 'REPORT_METRICS {"train_step": 600, "train_ce": 8}'})
    with pytest.raises(ValueError, match='Conflicting'):
        beaker_history_rows(lines)


def test_discovery_uses_published_links_and_caches_terminal_beaker_fallback(tmp_path):
    lines = [{'message': 'REPORT_METRICS ' + json.dumps({'train_step': r['_step'],
             'train_ce': r['train/CE loss']})} for r in curve()]
    calls = []
    job = SimpleNamespace(id='job', status=SimpleNamespace(HasField=lambda field: True))
    b = SimpleNamespace(workload=SimpleNamespace(get=lambda wid: SimpleNamespace(
        experiment=SimpleNamespace(tasks=['task']))), job=SimpleNamespace(
        list=lambda **kwargs: [job], logs=lambda *args, **kwargs: (
            calls.append('logs') or [SimpleNamespace(message=l['message'].encode()) for l in lines])))
    node = {'status': 'succeeded', 'receipt': {'workload_id': 'workload'},
            'summary': {'config': {'training': {'report_to': 'none'}}}}
    discover_beaker_history(node, b, tmp_path, 'now')
    assert node['wandb_discovery']['status'] == 'disabled'
    assert 'reported_metrics' not in node
    discover_beaker_history(node, b, tmp_path, 'later')
    assert calls == ['logs']
    review_node(node, None, tmp_path, 'now')
    assert invalid_result(node)
    assert logged_wandb_links([{'message': 'wandb: View run at https://forge.coreweave.com/wandb/ai2-llm/tokenizer-scaling/runs/real123?nw=1'}]) == {
        'ai2-llm/tokenizer-scaling/real123': 'https://forge.coreweave.com/wandb/ai2-llm/tokenizer-scaling/runs/real123'}
    assert logged_wandb_links([{'message': 'https://beaker.org/ex/workload'}]) == {}


def test_controller_stops_only_current_unhealthy_training_and_preserves_children(monkeypatch, tmp_path):
    from beaker import Beaker, BeakerWorkloadStatus
    from google.protobuf import json_format
    from unittest.mock import MagicMock
    s = fixture(); key = 'train-small-5gb-32000'; n = s['nodes'][key]
    n.update(status='running', receipt={'workload_id': 'current'}, chain_evaluation=True,
             chain_job_status='running', chain_snapshot={'current_stage':'train-model','summaries':{}},
             loss_curve_review={'status':'unhealthy','sources':[{'history_sha256':'evidence'}]},
             previous_attempts=[{'receipt':{'workload_id':'old'}}])
    child = study.eval_key(key)
    s['nodes'][child].update(chain_parent=key, receipt={'workload_id':'current'})
    s['nodes']['validation-live'] = {'stage':'validate-model','status':'running',
                                    'receipt':{'workload_id':'validation'}}
    gray = 'train-small-5gb-64000'; s['nodes'][gray].update(status='running',
        loss_curve_review={'status':'review_needed'})
    allowed = 'train-small-5gb-128000'; s['nodes'][allowed].update(status='running',
        loss_curve_review={'status':'unhealthy'},loss_health_decision={'status':'valid'})
    s['study']['stop_unhealthy_training'] = {'enabled':True}
    b = MagicMock(); b.__enter__.return_value = b
    monkeypatch.setattr(Beaker, 'from_env', lambda **kw:b)
    workload = lambda status:SimpleNamespace(experiment=SimpleNamespace(id='current'),
                                            status=BeakerWorkloadStatus[status].value)
    b.workload.get.side_effect = [workload('running'),workload('canceled')]
    b.workload.get_latest_job.return_value = SimpleNamespace(id='job')
    monkeypatch.setattr(json_format,'MessageToDict',lambda job:{'status':{'status':'STATUS_CANCELED'}})
    path=tmp_path/'state.json'; persisted=[]
    def cancel(w):
        persisted.append(json.loads(path.read_text())['nodes'][key]['unhealthy_stop'])
        return iter(['current'])
    b.workload.cancel.side_effect=cancel
    untouched={k:copy.deepcopy(s['nodes'][k]) for k in ('validation-live',gray,allowed)}
    receipt=copy.deepcopy(n['receipt']); history=copy.deepcopy(n['previous_attempts'])
    campaign.stop_unhealthy_training(s,path)
    assert n['status']=='canceled' and n['chain_job_status']=='canceled'
    assert n['unhealthy_stop']['confirmed_at'] and persisted[0]['loss_curve_review']['status']=='unhealthy'
    assert n['receipt']==receipt and n['previous_attempts']==history
    assert s['nodes'][child]['unhealthy_stop_parent']==key and s['nodes'][child]['hold_reason']
    assert all(s['nodes'][k]==v for k,v in untouched.items())
    assert b.workload.get.call_args_list[0].args==('current',)
    campaign.stop_unhealthy_training(s,path)
    assert b.workload.cancel.call_count==1
    import pytest
    with pytest.raises(ValueError,match='new explicit user authorization'):
        campaign.retry_node(s,key)
    data=report.build_report_data({**s,'nodes':{k:v for k,v in s['nodes'].items() if k!='validation-live'}})
    row=next(r for r in data['runs'] if r['id']==key)
    assert row['status']=='unhealthy' and row['training_status']=='canceled'
    assert all(v is None for v in row['metrics'].values())
    assert row['unhealthy_stop']['workload_id']=='current'


def test_controller_stop_requires_authorization_and_matching_receipt(monkeypatch,tmp_path):
    from beaker import Beaker, BeakerWorkloadStatus
    from unittest.mock import MagicMock
    s=fixture();n=s['nodes']['train-small-5gb-32000']
    n.update(status='running',loss_curve_review={'status':'unhealthy'},receipt={'workload_id':'current'})
    b=MagicMock();b.__enter__.return_value=b
    monkeypatch.setattr(Beaker,'from_env',lambda **kw:b)
    campaign.stop_unhealthy_training(s,tmp_path/'state.json')
    b.workload.get.assert_not_called()
    s['study']['stop_unhealthy_training']={'enabled':True}
    b.workload.get.return_value=SimpleNamespace(experiment=SimpleNamespace(id='wrong'),status=BeakerWorkloadStatus.running.value)
    campaign.stop_unhealthy_training(s,tmp_path/'state.json')
    b.workload.cancel.assert_not_called()
    assert n['unhealthy_stop_error'] and n['status']=='running'


def test_native_guard_matches_auditor_and_stops_before_recovery(tmp_path):
    from scripts.training_callbacks import LossHealthCallback
    for count,spike,stop_step in [(5,5.43,404),(3,10.,402),(2,5.43,None)]:
        cb=LossHealthCallback(str(tmp_path/f'{count}-{spike}'),warmup_steps=192)
        calls=[];cb.trainer=SimpleNamespace(cancel_run=calls.append)
        rows=curve(spike,count)
        for r in rows:cb.log_metrics(r['_step'],{'train/CE loss':r['train/CE loss']})
        expected=analyze_history(rows)['status']=='unhealthy'
        assert bool(calls)==expected
        if expected:
            record=json.loads((Path(cb.output_dir)/'health_stop.json').read_text())
            assert record['step']==stop_step and record['classification']=='unhealthy'
            assert record['loss_curve_evidence']['policy']['version']==2
            assert invalid_result({'health_stop':record,'loss_curve_review':{'status':'unavailable'}})
        else:assert not cb.stopped
