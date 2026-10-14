"""Published streams can release matching consumers while other streams are built."""
import copy
import json
import campaign


def fixture():
    corpus={'sha256':'raw-hash','documents':10,'text_bytes':2000}
    stream={'recipe':{'validation_id':'dolma100m-v1','documents_sha256':'raw-hash',
                      'pool':'5GB','vocab_size':32000},'documents':10,'bytes':2000}
    preparation={'stage':'prepare-validation','status':'running','receipt':{'workload_id':'current'}}
    campaign.record_validation_publications(preparation,[{'message':'VALIDATION_READY '+json.dumps(corpus)},
        {'message':'VALIDATION_TOKENS_READY '+json.dumps(stream)}])
    model={'stage':'train-model','status':'pending','arguments':[],'dependencies':['prepare-validation','tokens'],
           'data_size':'5GB','vocab_size':32000}
    return {'study':{'validation_node':'prepare-validation','validation_id':'dolma100m-v1',
                     'validation_overlap':{'enabled':True}},
            'nodes':{'prepare-validation':preparation,'tokens':{'status':'succeeded'},'train':model}}


def test_matching_stream_releases_before_whole_preparation_finishes():
    s=fixture()
    assert campaign.ready_nodes(s)==['train']
    assert s['nodes']['prepare-validation']['status']=='running'
    s['nodes']['train']['vocab_size']=64000
    assert campaign.ready_nodes(s)==[]
    s['nodes']['train']['vocab_size']=32000;s['nodes']['tokens']['status']='running'
    assert campaign.ready_nodes(s)==[]  # Other dependency gates remain enforced.


def test_no_release_without_permission_or_for_failure_or_stale_publication():
    for mode in ('disabled','failed','stale','mismatch','monitor_error'):
        s=fixture();p=s['nodes']['prepare-validation']
        if mode=='disabled':s['study']['validation_overlap']['enabled']=False
        elif mode=='failed':p['status']='failed'
        elif mode=='stale':p['receipt']['workload_id']='replacement'
        elif mode=='mismatch':p['validation_publications']['corpus']['sha256']='different'
        else:p['monitor_error']='unavailable'
        assert campaign.ready_nodes(s)==[],mode


def test_backfill_consumers_release_and_publications_survive_log_tails():
    s=fixture();s['nodes']['train']['stage']='validate-model'
    p=s['nodes']['prepare-validation'];saved=copy.deepcopy(p['validation_publications'])
    campaign.record_validation_publications(p,[{'message':'still tokenizing another stream'}])
    assert p['validation_publications']==saved and campaign.ready_nodes(s)==['train']
    p['receipt']['workload_id']='replacement'
    campaign.record_validation_publications(p,[{'message':'starting'}])
    assert campaign.ready_nodes(s)==[]
