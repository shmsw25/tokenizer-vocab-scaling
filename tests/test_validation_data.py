import hashlib
import json
from pathlib import Path
import random
import sys

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pipeline import read_json, write_json, raw_signature, digest
from validation_data import prepare_corpus, tokenize_validation, validation_root


def fixture(root):
    full=root/'raw_text/250GB/dolma3_250gb.jsonl';full.parent.mkdir(parents=True)
    rows=[{'text':f'Unique document {i if i<800 else i-800} café 한국어 payload'} for i in range(1000)]
    full.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    rng=random.Random(42);needed=400;parent_rows=[]
    for i,row in enumerate(rows):
        if needed and rng.randrange(1000-i)<needed:parent_rows.append(row);needed-=1
    parent=root/'raw_text/100GB/dolma3_100gb.jsonl';parent.parent.mkdir(parents=True)
    parent.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in parent_rows))
    write_json(root/'raw_text/provenance.json',dict(full=raw_signature(full),sample=raw_signature(parent),sampling_seed=42,sample_fraction=.4,repetitions=1))
    write_json(root/'raw_text/heldout/dolma3_100gb_shared_1000.json',dict(source_records=400,source_sha256=digest(parent)))
    tok=Tokenizer(models.WordLevel({'<|endoftext|>':0,'[UNK]':1},unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.Whitespace()
    ref=root/'tokenized/default_5GB_32000/tokenizer.json';ref.parent.mkdir(parents=True);tok.save(str(ref))
    write_json(ref.with_name('meta.json'),dict(train=dict(tokens=4000,documents=400)))
    path=root/'tokenizers/default_5GB_2/tokenizer.json';path.parent.mkdir(parents=True);tok.save(str(path))
    return full,parent,rows,parent_rows


def test_shared_validation_is_disjoint_deterministic_and_whole_documents(tmp_path):
    _,_,_,parent=fixture(tmp_path)
    info=prepare_corpus(tmp_path,target_tokens=100,seed=13,candidate_count=600)
    rows=[json.loads(l) for l in Path(info['output']).read_text().splitlines()]
    assert not {r['text'] for r in rows}&{r['text'] for r in parent}
    assert len({r['text_sha256'] for r in rows})==len(rows)
    assert info['candidate_parent_text_overlaps_removed']>0
    assert info['reference_tokens']>=100
    assert info['text_bytes']==sum(len(r['text'].encode()) for r in rows)
    assert info['sha256']==digest(Path(info['output']))
    assert prepare_corpus(tmp_path,target_tokens=100,seed=13,candidate_count=600)==info
    tokens=tokenize_validation(tmp_path,'5GB',2)
    assert tokens['tokens']==info['reference_tokens']
    assert tokens['documents']==info['documents']
    assert tokenize_validation(tmp_path,'5GB',2)==tokens
    second=tmp_path/'independent';fixture(second)
    other=prepare_corpus(second,target_tokens=100,seed=13,candidate_count=600)
    assert info['sha256']==other['sha256']


def test_validation_rejects_mutation_and_preserves_partial(tmp_path):
    fixture(tmp_path)
    p=validation_root(tmp_path)/'documents.jsonl.part';p.parent.mkdir(parents=True);p.write_text('keep')
    with pytest.raises(FileExistsError,match='Preserve'):prepare_corpus(tmp_path,target_tokens=100)
    assert p.read_text()=='keep'
    p.unlink();info=prepare_corpus(tmp_path,target_tokens=100)
    Path(info['output']).write_text('changed')
    with pytest.raises(ValueError,match='changed'):prepare_corpus(tmp_path,target_tokens=100)


def test_validation_checks_parent_reconstruction(tmp_path):
    fixture(tmp_path)
    p=tmp_path/'raw_text/heldout/dolma3_100gb_shared_1000.json';x=read_json(p);x['source_sha256']='wrong';write_json(p,x)
    with pytest.raises(ValueError,match='Reconstructed parent'):prepare_corpus(tmp_path,target_tokens=100)


def test_validation_stage_requests_cpu_and_mount():
    import submit
    args=submit.parser().parse_args(['prepare-validation','--cluster','ai2/saturn','ai2/neptune','--allow-gpu-reservation','--min-runtime','3h'])
    task=submit.make_spec(args)['tasks'][0]
    assert task['resources']['gpuCount']==0
    assert {'name':'CUDA_VISIBLE_DEVICES','value':''} in task['envVars']
    assert task['context']['minRuntime']=='3h'
    assert task['datasets'][1]['source']=={'weka':'oe-training-default'}


def test_scoring_windows_predict_every_token_once():
    from score_validation import validation_windows
    for count in (2,7,8,9,100):
        values=list(range(count));windows=list(validation_windows(values,7))
        assert [t for _,w in windows for t in w[1:]]==values[1:]
        assert all(2<=len(w)<=8 for _,w in windows)


def test_validation_scorer_matches_full_uniform_cross_entropy():
    import math
    import torch
    from types import SimpleNamespace
    from score_validation import score_stream
    class Uniform(torch.nn.Module):
        def forward(self,x):return SimpleNamespace(logits=torch.zeros((*x.shape,5)))
    values=[i%5 for i in range(103)]
    nll,count=score_stream(Uniform(),values,7,'cpu')
    assert count==102
    assert nll/count==pytest.approx(math.log(5),rel=1e-6)


def test_validation_stream_rejects_changed_tokenizer(tmp_path):
    from validation_data import validation_metadata
    fixture(tmp_path);prepare_corpus(tmp_path,target_tokens=100);tokenize_validation(tmp_path,'5GB',2)
    assert validation_metadata(tmp_path,'5GB',2)['tokens']>=100
    p=tmp_path/'tokenizers/default_5GB_2/tokenizer.json';p.write_text('{}')
    with pytest.raises(ValueError,match='identity'):validation_metadata(tmp_path,'5GB',2)
