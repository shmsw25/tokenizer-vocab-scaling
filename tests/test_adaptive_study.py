import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import adaptive_study as study
import campaign
import pipeline
import submit


def fixture():
    state = {'campaign': 'test', 'workspace': 'ai2/flex2', 'owner': 'sewonm', 'complete': False,
             'nodes': campaign.make_nodes([32000, 64000, 128000], list(study.MODELS)),
             'pool_sizes': list(pipeline.POOL_SIZES), 'training_pools': list(pipeline.TRAINING_POOLS)}
    for k, n in state['nodes'].items():
        if n['stage'] in ('sample-pool', 'download-data', 'train-tokenizer', 'tokenize-data'):
            n['status'] = 'succeeded'
        if n['stage'] == 'train-model' and n['data_size'] == '5GB':
            n.update(status='succeeded', receipt={'workload_id': k}, summary={
                'training': {'latest_metrics': {'validation_bpb': 1.0}}})
    study.initialize(state)
    return state


def finish(state, key, loss):
    state['nodes'][key].update(status='succeeded', summary={'training': {'latest_metrics': {'validation_bpb': loss}}})


def descent_fixture():
    s = fixture()
    gate_key = 'train-small-5gb-32000'
    s['nodes'][gate_key].update(status='running', chain_job_status='running')
    s['study']['small_10gb_lr_descent'] = {'enabled': True, 'gate': {
        'allocations': [{'key': gate_key, 'workload_id': gate_key}]}}
    s['study']['pool_lr_search'] = {'pools': ['10GB', '20GB'], 'learning_rates': [.001]}
    s['study']['validation_node'] = 'prepare-validation'
    s['nodes']['prepare-validation'] = {'stage': 'prepare-validation', 'status': 'succeeded',
                                      'arguments': [], 'dependencies': []}
    return s, gate_key


def test_small_10gb_descent_waits_for_exact_running_allocations_and_preserves_recipe():
    s, gate_key = descent_fixture()
    required = set()
    study.advance_small_10gb_lr_descent(s, required)
    cohort = s['study']['small_10gb_lr_descent']
    keys = [runs[0] for runs in cohort['runs'].values()]
    campaign.prepare_chains(s)
    assert len(keys) == 3
    assert not set(keys) & set(campaign.ready_nodes(s))
    # Finishing training alone is insufficient: the shared allocation is live.
    s['nodes'][gate_key]['status'] = 'succeeded'
    study.advance_small_10gb_lr_descent(s, required)
    assert not cohort['gate'].get('released_at')
    s['nodes'][gate_key].pop('chain_job_status')
    assert not set(keys) & set(campaign.ready_nodes(s))
    s['nodes'][gate_key]['chain_job_status'] = 'canceled'
    s['nodes'][gate_key]['receipt']['workload_id'] = 'different-replacement'
    assert not set(keys) & set(campaign.ready_nodes(s))
    s['nodes'][gate_key]['receipt']['workload_id'] = gate_key
    s['nodes'][gate_key]['monitor_error'] = 'inventory unavailable'
    assert not set(keys) & set(campaign.ready_nodes(s))
    s['nodes'][gate_key].pop('monitor_error')
    study.advance_small_10gb_lr_descent(s, required)
    assert cohort['gate']['released_at']
    assert set(keys) <= set(campaign.ready_nodes(s))
    for key in keys:
        n = s['nodes'][key]
        assert n['model_size'] == 'small' and n['data_size'] == '10GB'
        assert study.arg(n, '--learning-rate') == '0.0005'
        assert study.arg(n, '--embedding-weight-decay', '.1') == '.1'
        assert study.arg(n, '--validation-id') == 'dolma100m-v1'
        assert n['arguments'][1:] == s['nodes'][study.eval_key(key)]['arguments'][1:]
        args = submit.parser().parse_args([*n['arguments'], *campaign.placement_arguments(n)])
        spec = submit.parse_spec(submit.make_spec(args)).to_json()['tasks'][0]
        assert spec['resources']['gpuCount'] == 8
        assert spec['context']['minRuntime'] == 10_800_000_000_000
        assert n['placement_clusters'] == ['ai2/jupiter', 'ai2/saturn', 'ai2/neptune']
        assert args.batch_size == 128 and args.micro_batch_size == 16
    # User holds remain effective after the start gate opens.
    s['nodes'][keys[0]]['hold_reason'] = 'User hold'
    assert keys[0] not in campaign.ready_nodes(s)


def test_small_10gb_fallback_is_per_vocab_unhealthy_terminal_and_bounded():
    s, gate_key = descent_fixture()
    s['nodes'][gate_key].update(status='succeeded', chain_job_status='succeeded')
    required = set()
    study.advance_small_10gb_lr_descent(s, required)
    cohort = s['study']['small_10gb_lr_descent']
    runs = cohort['runs']['10GB:small:32000']
    node = s['nodes'][runs[0]]
    node.update(status='failed', receipt={'workload_id': 'initial'})
    # Setup failures, review flags, unresolved failures and ongoing stops do not trigger.
    node['loss_curve_review'] = {'status': 'review_needed'}
    study.advance_small_10gb_lr_descent(s, required)
    assert len(runs) == 1
    node['loss_health_decision'] = {'status': 'unhealthy'}
    node['failure_investigation'] = {'status': 'unresolved'}
    study.advance_small_10gb_lr_descent(s, required)
    assert len(runs) == 1
    node.pop('failure_investigation')
    node['chain_job_status'] = 'stopping'
    study.advance_small_10gb_lr_descent(s, required)
    assert len(runs) == 1
    node['chain_job_status'] = 'canceled'
    node['user_hold'] = True
    study.advance_small_10gb_lr_descent(s, required)
    assert len(runs) == 1
    node.pop('user_hold')
    study.advance_small_10gb_lr_descent(s, required)
    assert len(runs) == 2
    assert study.arg(s['nodes'][runs[-1]], '--learning-rate') == '0.0002'
    assert study.eval_key(runs[0]) not in required
    assert study.eval_key(runs[-1]) in required
    assert len(cohort['runs']['10GB:small:64000']) == 1
    assert study.training_gate_ready(s, s['nodes'][runs[-1]])
    node['loss_health_decision']['status'] = 'valid'
    assert not study.training_gate_ready(s, s['nodes'][runs[-1]])
    node['loss_health_decision']['status'] = 'unhealthy'
    for rate in (.0002, .0001):
        key = runs[-1]
        assert float(study.arg(s['nodes'][key], '--learning-rate')) == rate
        s['nodes'][key].update(status='canceled', chain_job_status='canceled',
            receipt={'workload_id': key}, loss_health_decision={'status': 'unhealthy'})
        study.advance_small_10gb_lr_descent(s, required)
    assert len(runs) == 3
    assert cohort['status']['10GB:small:32000'] == 'exhausted_unhealthy'
    before = len(s['nodes'])
    study.advance_small_10gb_lr_descent(s, required)
    assert len(s['nodes']) == before
    assert all(study.eval_key(k) not in required for k in runs)


def test_lower_pool_candidates_do_not_release_old_holds_or_expand_other_coordinates():
    s, gate_key = descent_fixture()
    s['nodes'][gate_key].update(status='succeeded', chain_job_status='succeeded')
    old = study.model_run(s, '10GB', 'small', 32000, .001, role='existing')
    s['nodes'][old]['hold_reason'] = 'User deferred'
    required = set()
    study.advance_small_10gb_lr_descent(s, required)
    key = s['study']['small_10gb_lr_descent']['runs']['10GB:small:32000'][0]
    finish(s, key, .8)
    s['nodes'][study.eval_key(key)]['status'] = 'succeeded'
    assert not study.advance_pool_lr_search(s, [32000, 64000, 128000], required)
    assert study.eval_key(old) in required
    assert s['nodes'][old]['hold_reason'] == 'User deferred'
    assert s['study']['pool_lr_search']['runs']['10GB:small:32000'] == [old, key]
    lower = [n for n in s['nodes'].values() if n.get('study_role') == 'small_10gb_lr_descent']
    assert len(lower) == 9 and all(n['model_size'] == 'small' and n['data_size'] == '10GB' for n in lower)


def test_lr_bracket_and_transfer_without_retuning():
    s = fixture();study.advance_plan(s)
    assert len([n for n in s['nodes'].values() if n.get('study_role') == 'lr_trial' and n['stage']=='train-model']) == 9
    b = s['study']['brackets']['small:32000']
    finish(s, b['lower'], .9);study.advance_plan(s)
    assert float(study.arg(s['nodes'][b['next']], '--learning-rate')) == .0005
    finish(s, b['next'], .95)
    other = s['study']['brackets']['medium:32000']
    finish(s, other['lower'], 1.1);study.advance_plan(s)
    assert float(study.arg(s['nodes'][other['next']], '--learning-rate')) == .004
    for pair, bracket in s['study']['brackets'].items():
        finish(s, bracket['lower'], .9)
    study.advance_plan(s)
    for bracket in s['study']['brackets'].values():finish(s, bracket['next'], .95)
    study.advance_plan(s)
    assert s['study']['selected']['small:32000']['lr'] == .001
    assert s['study']['phase'] == 'transfer_10gb_20gb'
    for coordinate,key in s['study']['transfers'].items():
        pair=':'.join(coordinate.split(':')[1:])
        assert float(study.arg(s['nodes'][key],'--learning-rate')) == s['study']['selected'][pair]['lr']
    before=len(s['nodes']);study.advance_plan(s);assert len(s['nodes'])==before
    assert all(s['nodes'][k].get('study_role')!='superseded_plan' for k in campaign.ready_nodes(s))


def test_retired_rate_and_explicit_five_gb_grid_do_not_expand_scope():
    s = fixture()
    s['study'].update(excluded_learning_rates=[.002, .004], fixed_learning_rates={'medium_large': .001},
        additional_learning_rates={'5GB': {'medium_large': [.0005]}},
        pool_min_runtime={'5GB': '1h'},
        five_gb_lr_search={'runs': {}},
        pool_lr_search={'pools': ['10GB', '20GB'], 'learning_rates': [.001, .002]})
    for model in study.MODELS:
        for vocab in (32000, 64000, 128000):
            pair = f'{model}:{vocab}'
            old = s['study']['brackets'][pair]['baseline']
            s['study']['selected'][pair] = {'key': old, 'lr': .002}
            base = study.model_run(s, '5GB', model, vocab, .001, role='existing')
            finish(s, base, 1.)
            s['nodes'][study.eval_key(base)]['status'] = 'succeeded'
            trials = [study.model_run(s, '5GB', model, vocab, .0005, role='explicit',
                                      embedding_weight_decay=wd) for wd in (.1, 0.)]
            for key in trials:
                assert s['nodes'][key]['min_runtime'] == '1h'
                args = submit.parser().parse_args([*s['nodes'][key]['arguments'],
                                                   *campaign.placement_arguments(s['nodes'][key])])
                task = submit.parse_spec(submit.make_spec(args)).to_json()['tasks'][0]
                assert task['context']['minRuntime'] == 3_600_000_000_000
                assert task['resources']['gpuCount'] == 8
            s['study']['five_gb_lr_search']['runs'][pair] = [base, *trials]
    study.advance_plan(s)
    assert s['study']['phase'] == 'pool_lr_search'
    assert all(v['lr'] != .002 for v in s['study']['selected'].values())
    assert all(not study.excluded_run(s, s['nodes'][k]) for k in s['study']['required'])
    assert all(not study.excluded_run(s, s['nodes'][k]) for k in campaign.ready_nodes(s))
    assert all(n['data_size'] == '5GB' for n in s['nodes'].values()
               if n['stage'] == 'train-model' and study.arg(n, '--learning-rate') == '0.0005')
    pair = 'medium_large:128000'
    zero = s['study']['five_gb_lr_search']['runs'][pair][-1]
    finish(s, zero, .8);s['nodes'][study.eval_key(zero)]['status'] = 'succeeded'
    study.advance_plan(s)
    assert s['study']['selected'][pair]['key'] == zero
    assert s['study']['selected'][pair]['embedding_weight_decay'] == 0
    count = len(s['nodes']);study.advance_plan(s);assert len(s['nodes']) == count
    with pytest.raises(ValueError, match='fixed'):
        study.model_run(s, '10GB', 'medium_large', 32000, .0005, role='unauthorized')
    with pytest.raises(ValueError, match='excluded'):
        study.model_run(s, '5GB', 'small', 32000, .002, role='retired')


def test_branch_requires_manifest_and_matching_chain_arguments():
    s=fixture()
    key=study.model_run(s,'100GB','medium_large',128000,.0005,role='100gb_trunk')
    with pytest.raises(ValueError,match='checkpoint'):
        study.model_run(s,'100GB','medium_large',128000,.0005,role='100gb_decay_branch',fraction=.05,parent=key)
    s['nodes'][key]['branch_checkpoints']={'0.05':{'parent':'/oe-training-default/example','step':900,'total_steps':1000}}
    branch=study.model_run(s,'100GB','medium_large',128000,.0005,role='100gb_decay_branch',fraction=.05,parent=key)
    for prefix in ('train','convert','eval'):
        n=s['nodes'][branch.replace('train-',prefix+'-',1)]
        a=submit.parser().parse_args([*n['arguments'],*campaign.placement_arguments(n)])
        pipeline.validate_args(a)
        assert a.micro_batch_size==4 and a.batch_size==128 and a.epoch_fraction==.05
        spec=submit.parse_spec(submit.make_spec(a)).to_json()['tasks'][0]
        assert spec['context']['minRuntime']==10_800_000_000_000
    assert s['nodes'][branch]['arguments'][1:]==s['nodes'][study.eval_key(branch)]['arguments'][1:]


def test_admission_counts_retained_allocations_and_fails_closed(tmp_path,monkeypatch):
    s=fixture();study.advance_plan(s)
    monkeypatch.setattr(study,'live_gpu_budget',lambda _:dict(reserved=96,assigned=88,limit=100))
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**k:pytest.fail('must not launch above cap'))
    campaign.submit_ready(s,tmp_path/'state.json')
    assert any(n.get('admission_wait') for n in s['nodes'].values())
    assert study.reservation(dict(expected_gpus=0,placement_clusters=['ai2/saturn']))==8
    monkeypatch.setattr(study,'live_gpu_budget',lambda _:(_ for _ in ()).throw(RuntimeError('offline')))
    campaign.submit_ready(s,tmp_path/'state.json')
    assert s['gpu_budget_error']


def test_gpu_inventory_reads_queued_min_runtime_and_counts_all_replicas(monkeypatch):
    from beaker import Beaker
    from google.protobuf import json_format
    from unittest.mock import MagicMock
    b = MagicMock()
    monkeypatch.setattr(Beaker, 'from_env', lambda **kwargs: b)
    b.__enter__.return_value = b
    queued = SimpleNamespace(experiment=SimpleNamespace(id='external-queued', tasks=[]),
                             environment=SimpleNamespace(id=''))
    live = SimpleNamespace(experiment=SimpleNamespace(id='chain-running', tasks=['train']),
                           environment=SimpleNamespace(id=''))
    b.workload.list.return_value = [queued, live]
    b.workload.get.side_effect = lambda identifier: queued if identifier == 'external-queued' else live
    b.job.list.return_value = [SimpleNamespace(id='live-job')]
    b.experiment.rpc_request.return_value.experiment_spec = '''tasks:
  - context: {minRuntime: 3h0m0s, priority: urgent}
    resources: {gpuCount: 8}
    replicas: 2
  - context: {minRuntime: 3h0m0s}
    resources: {gpuCount: 0}
'''
    monkeypatch.setattr(json_format, 'MessageToDict', lambda job: {
        'containerSpec': {'resourceRequest': {'gpuCount': 8}},
        'assignmentDetails': {'resourceAssignment': {'gpus': list(range(8))}},
    })
    s = {'workspace': 'ai2/test', 'owner': 'sewonm', 'study': {'gpu_scope': 'all_user', 'gpu_limit': 100},
         'nodes': {'train': {'receipt': {'workload_id': 'chain-running'}, 'status': 'succeeded',
                             'chain_job_status': 'running', 'expected_gpus': 8},
                   'eval': {'receipt': {'workload_id': 'chain-running'}, 'status': 'running',
                            'chain_parent': 'train', 'expected_gpus': 1}}}
    budget = study.live_gpu_budget(s)
    assert budget['reserved'] == 32  # Two queued replicas, CPU reservation, retained chain allocation.
    assert budget['assigned'] == 8
    assert len(budget['entries']) == 2
    b.experiment.get_spec.assert_not_called()
    b.experiment.rpc_request.side_effect = RuntimeError('inventory unavailable')
    with pytest.raises(RuntimeError, match='inventory unavailable'):
        study.live_gpu_budget(s)


def test_loss_health_guard_waits_for_sustained_deterioration(tmp_path):
    from scripts.training_callbacks import LossHealthCallback
    calls=[];cb=LossHealthCallback(str(tmp_path),warmup_steps=10)
    cb.trainer=SimpleNamespace(cancel_run=calls.append)
    for step in range(110, 610, 10):cb.log_metrics(step,{'train/CE loss':2.})
    cb.log_metrics(610,{'train/CE loss':10.})
    assert not calls
    for step in range(620, 1800, 10):cb.log_metrics(step,{'train/CE loss':5.})
    assert not calls  # Not yet 1000 optimizer steps above threshold.
    for step in range(1800, 2000, 10):cb.log_metrics(step,{'train/CE loss':5.})
    assert calls and (tmp_path/'health_stop.json').exists()


def test_loss_guard_does_not_stop_slow_recovery_or_moderate_spikes(tmp_path):
    from scripts.training_callbacks import LossHealthCallback
    calls=[];cb=LossHealthCallback(str(tmp_path),warmup_steps=10)
    cb.trainer=SimpleNamespace(cancel_run=calls.append)
    for step in range(110, 610, 10):cb.log_metrics(step,{'train/CE loss':2.})
    for step in range(610, 2610, 10):cb.log_metrics(step,{'train/CE loss':3.9})
    assert not calls
    # Loss remains >2x baseline for far longer than patience, but steadily recovers.
    for step in range(2610, 6610, 10):
        cb.log_metrics(step,{'train/CE loss':12 - (step-2610)*.001})
    assert not calls
    saved=cb.state_dict();resumed=LossHealthCallback(str(tmp_path),warmup_steps=10)
    resumed.load_state_dict(saved)
    assert resumed.best==2. and resumed.values==cb.values
    resumed.trainer=SimpleNamespace(global_step=6600,cancel_run=calls.append)
    resumed.pre_train()
    assert resumed.high_since is None and resumed.last_recovery_step==6600


@pytest.mark.parametrize('loss',[float('nan'),float('inf')])
def test_loss_guard_still_stops_nonfinite_loss(tmp_path,loss):
    from scripts.training_callbacks import LossHealthCallback
    calls=[];cb=LossHealthCallback(str(tmp_path),warmup_steps=192)
    cb.trainer=SimpleNamespace(cancel_run=calls.append)
    cb.log_metrics(1,{'train/CE loss':loss})
    assert calls==['Non-finite training CE loss']


def test_explicit_larger_pool_lr_search_reuses_runs_and_gates_100gb():
    s=fixture();study.advance_plan(s)
    for b in s['study']['brackets'].values():finish(s,b['lower'],.9)
    study.advance_plan(s)
    for b in s['study']['brackets'].values():finish(s,b['next'],.95)
    study.advance_plan(s)
    old=dict(s['study']['transfers'])
    s['study']['pool_lr_search']={'pools':['10GB','20GB'],'learning_rates':[.001,.002]}
    study.advance_plan(s)
    search=s['study']['pool_lr_search'];keys=[key for pair in search['runs'].values() for key in pair]
    assert len(keys)==len(set(keys))==36
    assert {key for key in old.values() if float(study.arg(s['nodes'][key],'--learning-rate')) in (.001,.002)}<=set(keys)
    assert s['study']['phase']=='pool_lr_search' and 'sample-100gb' not in s['nodes']
    count=len(s['nodes']);study.advance_plan(s);assert len(s['nodes'])==count
    for key in keys:
        finish(s,key,.8 if float(study.arg(s['nodes'][key],'--learning-rate'))==.002 else .9)
        s['nodes'][study.eval_key(key)]['status']='succeeded'
    study.advance_plan(s)
    assert all(choice['lr']==.002 for choice in search['selected'].values())
    assert s['study']['phase']=='100gb_horizons'
    # 100GB continues using the 5GB-selected LR; larger-pool tuning is separate.
    assert all(float(study.arg(s['nodes'][key],'--learning-rate'))==s['study']['selected'][pair]['lr']
               for pair,key in s['study']['hundred_gb'].items())


def test_health_resume_preserves_lr_and_history_and_clears_stale_health():
    s=fixture();study.advance_plan(s)
    key=s['study']['brackets']['small:32000']['lower'];node=s['nodes'][key]
    node.update(status='failed',chain_job_status='failed',receipt={'workload_id':'old'},
                health_stop={'reason':'old guard'},wandb_final_checked=True)
    campaign.prepare_chains(s)
    with pytest.raises(ValueError,match='authorized'):campaign.retry_node(s,key)
    original_lr=study.arg(node,'--learning-rate')
    node['arguments'].append('--resume-latest')
    campaign.retry_node(s,key)
    assert node['status']=='pending' and 'health_stop' not in node and 'wandb_final_checked' not in node
    assert node['previous_attempts'][-1]['health_stop']['reason']=='old guard'
    assert study.arg(node,'--learning-rate')==original_lr
    assert 'receipt' not in s['nodes'][study.eval_key(key)]
    assert campaign.ready_nodes(s)[0]==key


def test_full_100gb_excludes_shared_holdout_without_touching_parent(tmp_path):
    from sample_pool import sample_pool
    parent=tmp_path/'raw_text/100GB/dolma3_100gb.jsonl';parent.parent.mkdir(parents=True)
    parent.write_text(''.join(json.dumps({'text':str(i)})+'\n' for i in range(100)))
    identity=pipeline.raw_signature(parent)
    pipeline.write_json(tmp_path/'raw_text/provenance.json',{'sample':identity})
    info=sample_pool(tmp_path,'100GB',heldout_count=10)
    hold=pipeline.read_json(Path(info['heldout_file']))
    excluded={d['text'] for d in hold['documents']}
    rows=[json.loads(l)['text'] for l in Path(info['output']['path']).read_text().splitlines()]
    assert len(rows)==90 and not excluded.intersection(rows)
    assert pipeline.raw_signature(parent)==identity


def test_unhealthy_trial_does_not_block_selection_or_required_evaluations():
    s=fixture();study.advance_plan(s)
    pair=s['study']['brackets']['small:32000'];lower=pair['lower']
    s['nodes'][lower].update(status='failed',health_stop={'reason':'Non-finite loss'})
    study.advance_plan(s)
    assert float(study.arg(s['nodes'][pair['next']], '--learning-rate'))==.004
    assert study.eval_key(lower) not in s['study']['required']


def test_native_nan_failure_is_unhealthy_without_callback_marker():
    node = {'stage': 'train-model', 'status': 'failed', 'receipt': {'workload_id': 'current'},
            'job_id': 'job', 'arguments': []}
    lines = [{'timestamp': '2026-10-14T14:50:41Z',
              'message': 'RuntimeError: nan loss encountered at step 485'}]
    campaign.record_native_loss_failure(node, lines)
    assert node['health_stop']['step'] == 485
    assert node['health_stop']['workload_id'] == 'current'
    assert study.completed_score(node) == float('inf')
    with pytest.raises(ValueError, match='explicitly authorized checkpoint resume'):
        campaign.retry_node({'nodes': {'trial': node}}, 'trial')


def test_unresolved_environment_failure_cannot_decide_pool_learning_rate():
    s = fixture()
    s['study']['pool_lr_search'] = {'pools': ['10GB'], 'learning_rates': [.001, .002]}
    study.advance_pool_lr_search(s, [32000], set())
    coordinate = '10GB:medium:32000'
    failed, healthy = s['study']['pool_lr_search']['runs'][coordinate]
    finish(s, healthy, .9)
    s['nodes'][failed].update(status='failed', health_stop={'reason': 'Non-finite loss'},
        failure_investigation={'status': 'unresolved', 'reason': 'Suspected faulty node'})
    study.advance_pool_lr_search(s, [32000], set())
    assert study.completed_score(s['nodes'][failed]) is None
    assert coordinate not in s['study']['pool_lr_search'].get('selected', {})
    # A completed investigation can restore ordinary unhealthy-trial handling.
    s['nodes'][failed]['failure_investigation']['status'] = 'resolved'
    assert study.completed_score(s['nodes'][failed]) == float('inf')


@pytest.mark.parametrize('stage,status,message', [
    ('train-model', 'running', 'RuntimeError: nan loss encountered at step 485'),
    ('train-model', 'succeeded', 'RuntimeError: nan loss encountered at step 485'),
    ('evaluate-model', 'failed', 'RuntimeError: nan loss encountered at step 485'),
    ('train-model', 'failed', 'RuntimeError: CUDA out of memory'),
])
def test_native_nan_classification_does_not_swallow_other_failures(stage, status, message):
    node = {'stage': stage, 'status': status, 'receipt': {'workload_id': 'current'}}
    campaign.record_native_loss_failure(node, [{'timestamp': 'now', 'message': message}])
    assert 'health_stop' not in node


def test_horizons_release_only_after_checkpoint_and_complete_only_after_all_evals():
    s=fixture();study.advance_plan(s)
    for b in s['study']['brackets'].values():finish(s,b['lower'],.9)
    study.advance_plan(s)
    for b in s['study']['brackets'].values():finish(s,b['next'],.95)
    study.advance_plan(s)
    for key in s['study']['required']:s['nodes'][key]['status']='succeeded'
    study.advance_plan(s)
    assert s['study']['phase']=='100gb_horizons' and 'sample-100gb' in s['nodes']
    assert not any(n.get('study_role')=='100gb_decay_branch' for n in s['nodes'].values())
    trunk=s['study']['hundred_gb']['small:32000']
    s['nodes'][trunk]['branch_checkpoints']={'0.05':{'parent':'/oe-training-default/run','step':900,'total_steps':1000}}
    study.advance_plan(s)
    branches=[k for k,n in s['nodes'].items() if n['stage']=='train-model' and n.get('study_role')=='100gb_decay_branch']
    assert len(branches)==1 and s['study']['phase']!='complete'
    # Successful trunks alone cannot silently skip missing fractional checkpoints.
    for key in s['study']['required']:s['nodes'][key]['status']='succeeded'
    study.advance_plan(s)
    assert s['study']['phase']=='100gb_horizons'


def test_corrected_architecture_never_reuses_legacy_and_waits_for_validation():
    s=fixture();s['study'].update(architecture_revision=2,validation_node='prepare-validation',architecture_archive_node='archive-medium-large')
    old=s['study']['brackets']['medium_large:32000']['baseline'];s['nodes'][old]['legacy_architecture']=True
    new=study.model_run(s,'5GB','medium_large',32000,.001,role='lr_baseline')
    assert new!=old and new.endswith('-prenorm-v2')
    assert '--resume-latest' not in s['nodes'][new]['arguments']
    assert '--validation-id' in s['nodes'][new]['arguments']
    assert {'prepare-validation','archive-medium-large'}<=set(s['nodes'][new]['dependencies'])
    for k in (old,new):s['nodes'][k].update(status='succeeded',summary={'model_dir':'/models/'+k,'training':{'latest_metrics':{'validation_bpb':1.}}})
    s['nodes'][study.eval_key(new)]['status']='succeeded'
    study.sync_shared_validation(s)
    assert study.completed_score(s['nodes'][new]) is None
    vk=new.replace('train-','validation-',1);s['nodes'][vk].update(status='succeeded',summary={'metrics':{'validation_bpb':.95}})
    study.sync_shared_validation(s)
    assert study.completed_score(s['nodes'][new])==.95
    assert old.replace('train-','validation-',1) not in s['nodes']
