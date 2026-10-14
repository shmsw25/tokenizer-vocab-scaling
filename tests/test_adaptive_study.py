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


def test_loss_health_guard_waits_for_sustained_deterioration(tmp_path):
    from scripts.training_callbacks import LossHealthCallback
    calls=[];cb=LossHealthCallback(str(tmp_path),warmup_steps=10)
    cb.trainer=SimpleNamespace(cancel_run=calls.append)
    for i in range(40):cb.log_metrics(110+i,{'train/CE loss':2.})
    cb.log_metrics(160,{'train/CE loss':10.})
    assert not calls
    for i in range(40):cb.log_metrics(161+i,{'train/CE loss':4.})
    assert calls and (tmp_path/'health_stop.json').exists()


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
