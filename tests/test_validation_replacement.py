"""Shared Dolma replacement, automatic evaluation, and the fixed architecture cohort."""
from pathlib import Path
from types import SimpleNamespace
import pytest

import adaptive_study as study
import pipeline
import report
import score_validation
import submit
from test_adaptive_study import fixture, finish
from test_report import state_fixture


def shared_score(checkpoint='model/step100'):
    return dict(identity=dict(validation_id='dolma100m-v1',checkpoint=checkpoint,
        exported='model/hf/step100',documents_sha256='shared-text-hash'),
        metrics={'validation_bpb':1.1,'validation_ce':3.2},documents=100000,
        text_bytes=400000000,stream_tokens=100000123,predicted_tokens=100000122,
        scoring='Repository byte-normalized CE')


def test_backfill_replaces_only_dolma_and_is_idempotent(tmp_path):
    old={'eval':'dolma_bpb','bpb':.7,'documents':1000}
    pipeline.write_json(tmp_path/'evaluation/dolma_bpb.json',old)
    pipeline.write_json(tmp_path/'evaluation_summary.json',dict(checkpoint='model/step100',
        metrics={'dolma_bpb':old,'blimp':{'accuracy':.8}}))
    result=shared_score();metric=score_validation.replace_dolma_result(tmp_path,result)
    summary=pipeline.read_json(tmp_path/'evaluation_summary.json')
    assert summary['metrics']=={'dolma_bpb':metric,'blimp':{'accuracy':.8}}
    assert summary['shared_validation']==result
    assert metric['bpb']==1.1 and metric['validation_id']=='dolma100m-v1'
    assert score_validation.replace_dolma_result(tmp_path,result)==metric
    history=list((tmp_path/'evaluation_history').glob('*.json'))
    assert len(history)==1 and pipeline.read_json(history[0])==old
    before=(tmp_path/'evaluation/dolma_bpb.json').read_bytes()
    with pytest.raises(ValueError,match='different checkpoint'):
        score_validation.replace_dolma_result(tmp_path,shared_score('other/step100'))
    assert (tmp_path/'evaluation/dolma_bpb.json').read_bytes()==before


def test_future_evaluation_scores_shared_corpus_even_with_cached_old_dolma(tmp_path,monkeypatch):
    old={'eval':'dolma_bpb','bpb':.7,'documents':1000}
    pipeline.write_json(tmp_path/'evaluation/dolma_bpb.json',old)
    pipeline.write_json(tmp_path/'training_summary.json',{'checkpoint':'model/step100'})
    pipeline.write_json(tmp_path/'conversion_summary.json',{})
    monkeypatch.setattr(pipeline,'model_artifacts',lambda a:(tmp_path,Path('model/step100'),None,Path('model/hf/step100')))
    monkeypatch.setattr(pipeline,'pool_provenance',lambda a:None)
    seen=[]
    def score(a):
        seen.append(a);return shared_score()
    monkeypatch.setattr(score_validation,'score_validation',score)
    args=SimpleNamespace(root=tmp_path,model_size='small',data_size='5GB',vocab_size=32000,evals=['dolma_bpb'])
    pipeline.evaluate_model(args)
    assert len(seen)==1 and seen[0].model_directory==tmp_path
    summary=pipeline.read_json(tmp_path/'evaluation_summary.json')
    assert summary['metrics']['dolma_bpb']['bpb']==1.1
    assert summary['shared_validation']['identity']['validation_id']=='dolma100m-v1'


def test_fixed_medium_large_rate_supersedes_brackets_and_pool_search():
    s=fixture();s['study']['fixed_learning_rates']={'medium_large':.001}
    s['study']['pool_lr_search']={'pools':['10GB','20GB'],'learning_rates':[.001,.002]}
    study.advance_plan(s)
    for pair,b in s['study']['brackets'].items():
        if pair.startswith('medium_large:'):
            assert b['fixed_lr']==.001 and 'lower' not in b and 'next' not in b
        else:finish(s,b['lower'],.9)
    study.advance_plan(s)
    for pair,b in s['study']['brackets'].items():
        if not pair.startswith('medium_large:'):finish(s,b['next'],.95)
    for coord,keys in s['study']['pool_lr_search']['runs'].items():
        assert len(keys)==(1 if ':medium_large:' in coord else 2)
        for key in keys:
            finish(s,key,.8);s['nodes'][study.eval_key(key)]['status']='succeeded'
    study.advance_plan(s)
    assert s['study']['phase']=='100gb_horizons'
    ml=[s['nodes'][k] for pair,k in s['study']['hundred_gb'].items() if pair.startswith('medium_large:')]
    assert len(ml)==3 and all(float(study.arg(n,'--learning-rate'))==.001 for n in ml)
    with pytest.raises(ValueError,match='user-fixed'):
        study.model_run(s,'5GB','medium_large',128000,.002,role='lr_trial')


def test_in_job_shared_score_avoids_duplicate_validation_job():
    s=fixture();s['study'].update(validation_node='prepare-validation',dolma_validation_id='dolma100m-v1')
    key='train-small-5gb-32000'
    s['nodes'][study.eval_key(key)].update(status='succeeded',summary={'shared_validation':shared_score()})
    study.sync_shared_validation(s)
    assert s['nodes'][key]['shared_validation']==shared_score()
    assert key.replace('train-','validation-',1) not in s['nodes']
    assert study.completed_score(s['nodes'][key])==1.1


def test_shared_dolma_replacement_never_falls_back_to_old_score():
    state=state_fixture();state['study']={'dolma_validation_id':'dolma100m-v1'}
    key='train-small-5gb-32000';n=state['nodes'][key]
    n.update(status='succeeded',shared_validation_required=True)
    state['nodes']['eval-small-5gb-32000']['summary']={'metrics':{
        'dolma_bpb':{'bpb':.7},'blimp':{'accuracy':.8}}}
    row=next(r for r in report.build_report_data(state)['runs'] if r['id']==key)
    assert row['metrics']['dolma_bpb'] is None and row['metrics']['blimp']==.8
    n['shared_validation']=shared_score()
    data=report.build_report_data(state)
    row=next(r for r in data['runs'] if r['id']==key)
    assert row['metrics']['dolma_bpb']==row['metrics']['validation_bpb']==1.1
    assert data['metric_labels']['dolma_bpb']=='Shared validation Dolma BPB ↓'


def test_default_training_and_downstream_validation_identity():
    for stage in pipeline.MODEL_STAGES:
        args=submit.parser().parse_args([stage,'--vocab-size','32000'])
        assert args.validation_id=='dolma100m-v1'
        arguments=submit.stage_arguments(args)
        assert arguments[arguments.index('--validation-id')+1]=='dolma100m-v1'


def test_one_time_script_cleanup_waits_for_every_backfill_target(tmp_path,monkeypatch):
    s=fixture();s['study'].update(validation_node='prepare-validation',dolma_validation_id='dolma100m-v1')
    key='train-small-5gb-32000';script=tmp_path/'.build/validation-replacement/migrate.py'
    script.parent.mkdir(parents=True);script.write_text('# one-time migration\n')
    s['study']['validation_backfill']={'targets':[key],'script_sha256':pipeline.digest(script),'status':'pending'}
    monkeypatch.setattr(pipeline,'REPO',tmp_path)
    study.sync_shared_validation(s);assert script.exists()
    s['nodes'][study.eval_key(key)].update(status='succeeded',summary={'shared_validation':shared_score()})
    study.sync_shared_validation(s)
    assert s['study']['validation_backfill']['status']=='complete' and not script.exists()
