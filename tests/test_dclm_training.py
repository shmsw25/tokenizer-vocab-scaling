import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import campaign
import dclm_data
import pipeline
import report
import submit
import validation_data
from dclm_training import authorize_small_10gb


def test_dclm_training_plan_checks_real_streams_and_separates_output(tmp_path, monkeypatch):
    raw = tmp_path / 'raw_text_dclm/10GB/dclm_10gb.jsonl'
    raw.parent.mkdir(parents=True)
    raw.write_text(''.join(json.dumps({'text': 'x'*10000})+'\n' for _ in range(100)))
    heldout = tmp_path / 'raw_text_dclm/heldout/sample.json'
    pipeline.write_json(heldout, {'fixture': True})
    pool = {'output_path': str(raw), 'output': {'bytes': raw.stat().st_size, 'sha256': pipeline.digest(raw)},
        'records': 100, 'text_bytes': 1000000,
        'recipe': {'heldout_file': str(heldout), 'heldout_sha256': pipeline.digest(heldout)}}
    pipeline.write_json(raw.with_name('pool.json'), pool)
    for vocab, tokens in [(32000, 700000), (64000, 630000)]:
        tok = dclm_data.tokenizer_path(tmp_path, '10GB', vocab)
        tok.parent.mkdir(parents=True);tok.write_text(f'fixture-tokenizer-{vocab}')
        directory = dclm_data.tokenized_path(tmp_path, '10GB', vocab);directory.mkdir(parents=True)
        (directory/'tokenizer.json').write_bytes(tok.read_bytes())
        meta = {'vocab_size': vocab, 'eos_token_id': 0, 'tokens_per_byte': tokens/1000000,
            'data_files': [str(raw)], 'seed':42, 'eval_fraction': .01,
            'train': {'documents':90, 'bytes':900000, 'tokens':tokens-10000},
            'eval': {'documents':10, 'bytes':100000, 'tokens':10000}}
        repack = {'dtype':'uint32', 'tokenizer_sha256':pipeline.digest(tok)}
        for split in ['train','eval']:
            np.zeros(meta[split]['tokens'], dtype=np.uint32).tofile(directory/f'{split}.bin')
            spans=directory/f'{split}.documents.csv.gz';spans.write_bytes(b'fixture-spans')
            repack[split] = {**meta[split], 'sha256':pipeline.digest(spans),
                'token_sha256':pipeline.digest(directory/f'{split}.bin')}
        pipeline.write_json(directory/'meta.json',meta)
        pipeline.write_json(directory/'repack.json',repack)
        pipeline.write_json(directory/'pipeline.json',{'recipe':{'pool':pool,'tokenizer_sha256':pipeline.digest(tok)}})
    seen=[]
    def metadata(root,pool,vocab,**kwargs):
        seen.append(kwargs['dataset']);return {'fixture':'shared-Dolma-with-DCLM-tokenizer'}
    monkeypatch.setattr(validation_data,'validation_metadata',metadata)
    args=submit.parser().parse_args(['train-model','--root',str(tmp_path),'--dataset','dclm',
        '--data-size','10GB','--vocab-size','64000','--learning-rate','.0005',
        '--weight-decay','.1','--embedding-weight-decay','.1'])
    pipeline.validate_args(args)
    cfg,out=pipeline.training_plan(args)
    assert 'models_dclm' in out.parts and cfg['pipeline']['dataset']=='dclm'
    assert cfg['seq_length']==round(4096*.9) and seen==['dclm']
    assert cfg['training']['learning_rate']==.0005 and cfg['training']['weight_decay']==.1
    assert cfg['training']['warmup_steps']==192 and cfg['training']['decay_fraction']==.1
    assert cfg['pipeline']['global_batch_size']==128
    for stage in ['convert-model','evaluate-model']:
        args.stage=stage
        assert pipeline.training_plan(args)==(cfg,out)
    with (pipeline.tokenized_dir(args)/'train.bin').open('r+b') as handle:handle.write(b'broken')
    with pytest.raises(ValueError,match='hash/count'):pipeline.training_plan(args)


def test_validation_dataset_namespaces_do_not_alias(tmp_path):
    from test_validation_data import fixture
    fixture(tmp_path)
    validation_data.prepare_corpus(tmp_path,target_tokens=100,seed=13,candidate_count=600)
    dolma=tmp_path/'tokenizers/default_5GB_2/tokenizer.json'
    dclm=tmp_path/'tokenizers_dclm/default_5GB_2/tokenizer.json'
    dclm.parent.mkdir(parents=True);dclm.write_bytes(dolma.read_bytes())
    first=validation_data.tokenize_validation(tmp_path,'5GB',2)
    second=validation_data.tokenize_validation(tmp_path,'5GB',2,dataset='dclm')
    assert first['output']!=second['output']
    assert first['sha256']==second['sha256']
    assert second['recipe']['tokenizer_dataset']=='dclm'
    assert validation_data.validation_metadata(tmp_path,'5GB',2)==first
    assert validation_data.validation_metadata(tmp_path,'5GB',2,dataset='dclm')==second
    pipeline.write_json(Path(second['output']).with_name('meta.json'),first)
    with pytest.raises(ValueError,match='identity'):validation_data.validation_metadata(tmp_path,'5GB',2,dataset='dclm')


def fixture_authorization(monkeypatch):
    monkeypatch.setattr(report,'dclm_training_tokens',lambda d:[{'pool':'10GB','tokens':{'32K':1,'64K':1,'128K':1}}])
    state={'study':{'root':'/weka','required':[],'validation_node':'prepare-validation',
        'training_submission_hold':{'reason':'User hold'}},
        'nodes':{'prepare-validation':{'stage':'prepare-validation','status':'succeeded'}}}
    dclm={'nodes':{f'dclm-tokens-10gb-{v}':{'stage':'tokenize-dclm-data','status':'succeeded',
        'arguments':[],'data_size':'10GB','vocab_size':v,'receipt':{'workload_id':str(v),'revision':'source'},
        'summary':{'stage':'tokenize-dclm-data','revision':'source'}} for v in [32000,64000,128000]}}
    return state,dclm


def test_authorization_is_exact_and_preserves_hold_chains_and_existing_receipts(monkeypatch):
    state,dclm=fixture_authorization(monkeypatch)
    original=copy.deepcopy(dclm)
    auth=authorize_small_10gb(state,dclm)
    assert dclm==original and state['study']['training_submission_hold']
    assert len(auth['runs'])==3 and len(auth['required'])==3
    assert all(state['nodes'][k]['stage']=='tokenize-validation' for k in campaign.ready_nodes(state))
    for key in auth['runs']:
        n=state['nodes'][key]
        state['nodes'][n['validation_stream_node']]['status']='succeeded'
        args=submit.parser().parse_args([*n['arguments'],*campaign.placement_arguments(n)])
        pipeline.validate_args(args)
        task=submit.make_spec(args)['tasks'][0]
        assert task['resources']['gpuCount']==8 and task['constraints']['cluster']==list(submit.GPU_CLUSTERS)
        assert task['context']=={'priority':'urgent','minRuntime':'3h'}
        assert '--dataset' in task['arguments'] and args.dataset=='dclm'
        assert {'name':'TOKENIZER_CHAIN_EVALUATION','value':'1'} in task['envVars']
        for prefix in ['convert','eval']:
            child=state['nodes'][key.replace('train-',prefix+'-',1)]
            assert child['chain_parent']==key and child['arguments'][1:]==n['arguments'][1:]
    assert set(campaign.ready_nodes(state))==set(auth['runs'])
    key=auth['runs'][0]
    other=copy.deepcopy(state['nodes'][key]);state['nodes']['unapproved']=other
    assert 'unapproved' not in campaign.ready_nodes(state)
    state['nodes'][key]['arguments']+=['--resume-latest']
    assert key not in campaign.ready_nodes(state)
    with pytest.raises(ValueError,match='already authorized'):authorize_small_10gb(state,dclm)


def test_report_separates_dclm_selection_and_compute(monkeypatch):
    from test_report import state_fixture,metadata
    import adaptive_study
    state=state_fixture();adaptive_study.initialize(state)
    old=state['nodes']['train-small-10gb-32000']
    old.update(status='succeeded',study_role='baseline',reported_metrics={'validation_bpb':2})
    old.pop('hold_reason',None)
    new=copy.deepcopy(old);new.update(dataset='dclm',reported_metrics={'validation_bpb':1})
    state['nodes']['train-dclm-new']=new
    state['nodes']['tokens-10gb-32000'].update(status='succeeded',summary={'meta':metadata()})
    state['nodes']['dclm-tokens']=dict(stage='tokenize-dclm-data',status='succeeded',vocab_size=32000,
        data_size='10GB',summary={'meta':metadata(20000000)})
    data=report.build_report_data(state)
    assert 'train-dclm-new' in data['selected'] and 'train-small-10gb-32000' in data['selected']
    row=next(r for r in data['runs'] if r['id']=='train-dclm-new')
    assert row['compute']['train_tokens']==20000000
    assert adaptive_study.model_run(state,'10GB','small',32000,.002,role='check')!='train-dclm-new'
    source=(pipeline.REPO/'scripts/report_template.html').read_text()
    assert source.index('id="dclm-run-grid"')<source.index('id="run-grid"')
