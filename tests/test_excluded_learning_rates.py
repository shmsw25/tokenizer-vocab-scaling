"""User-retired LR trials cannot return through planning, retry, or backfill."""
import copy
import pytest
import adaptive_study as study
import campaign
import report
from test_adaptive_study import fixture, finish
from test_report import state_fixture


def test_excluded_higher_trial_selects_remaining_rates_and_advances():
    s=fixture();s['study']['excluded_learning_rates']=[.004]
    study.advance_plan(s)
    for bracket in s['study']['brackets'].values():finish(s,bracket['lower'],1.1)
    study.advance_plan(s)
    for pair,bracket in s['study']['brackets'].items():
        if pair.startswith(('small:', 'medium:')):
            assert bracket['skipped_next_lr']==.004 and 'next' not in bracket
            assert s['study']['selected'][pair]['lr']==.002
        else:finish(s,bracket['next'],.95)
    study.advance_plan(s)
    assert s['study']['phase']=='transfer_10gb_20gb'
    assert not any(study.excluded_run(s,n) for n in s['nodes'].values())
    before=len(s['nodes']);study.advance_plan(s);assert len(s['nodes'])==before
    with pytest.raises(ValueError,match='excluded'):
        study.model_run(s,'100GB','small',32000,.004,role='100gb_trunk')


def test_existing_completed_excluded_trial_is_not_selected_or_rescored():
    s=fixture();study.advance_plan(s);b=s['study']['brackets']['small:32000']
    finish(s,b['lower'],1.1);study.advance_plan(s);retired=b['next']
    finish(s,retired,.5);study.advance_plan(s)
    assert s['study']['selected']['small:32000']['key']==retired
    s['study'].update(excluded_learning_rates=[.004],validation_node='prepare-validation',
        validation_backfill={'targets':[retired]})
    for kind,score in [('baseline',1.0),('lower',1.1)]:
        s['nodes'][b[kind]]['shared_validation']={'metrics':{'validation_bpb':score}}
    study.advance_plan(s)
    assert retired not in s['study']['selected']['small:32000']['candidates']
    assert study.eval_key(retired) not in s['study']['required']
    assert retired.replace('train-','validation-',1) not in s['nodes']
    assert s['study']['validation_backfill']['skipped_excluded']==[retired]
    assert s['study']['validation_backfill']['status']=='complete'
    assert s['nodes'][retired]['status']=='succeeded'  # Historical evidence is intact.
    with pytest.raises(ValueError,match='excluded'):campaign.retry_node(s,retired)
    for n in s['nodes'].values():n['status']='succeeded'
    s['nodes'][retired]['status']='pending'
    assert retired not in campaign.ready_nodes(s)


def test_report_hides_excluded_completed_trial_and_keeps_other_rates():
    s=state_fixture();s['study']={'excluded_learning_rates':[.004]}
    source=s['nodes']['train-small-5gb-32000'];retired=copy.deepcopy(source)
    a=retired['arguments'];a[a.index('--learning-rate')+1]='0.004'
    retired['status']='succeeded';s['nodes']['train-small-5gb-32000-lr0p004']=retired
    data=report.build_report_data(s)
    assert all(r['lr']!=.004 for r in data['runs'])
    assert any(r['id']=='train-small-5gb-32000' for r in data['runs'])
