import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import campaign
import dolma2_comparison as comparison
import pipeline
import submit
import validation_data
from test_emo_trials import fixture


def test_two_exact_admissions_preserve_holds_and_share_cpu_preparation(monkeypatch):
    state = fixture()
    original = copy.deepcopy(state)
    auth = comparison.authorize_two_runs(state)
    assert len(auth['runs']) == len(auth['required']) == 2
    assert len(state['nodes']) == len(original['nodes']) + 7
    assert all(state['nodes'][key] == node for key, node in original['nodes'].items())
    assert state['study']['training_submission_hold'] == original['study']['training_submission_hold']
    prep = auth['preparation_node']
    assert campaign.ready_nodes(state) == [prep]
    for key in [prep, *auth['runs']]:
        n = state['nodes'][key]
        args = submit.parser().parse_args([*n['arguments'], *campaign.placement_arguments(n)])
        pipeline.validate_args(args)
        spec = submit.make_spec(args)
        submit.parse_spec(spec)  # Exercise the Beaker schema for this dedicated stage.
        task = spec['tasks'][0]
        assert args.tokenizer_source == 'dolma2' and args.vocab_size == 100278
        assert '--tokenizer-source' in task['arguments']
        if key == prep:
            assert task['resources']['gpuCount'] == 0
            assert {'name': 'CUDA_VISIBLE_DEVICES', 'value': ''} in task['envVars']
        else:
            assert args.training_recipe == 'emo-dense-v1' and args.batch_size == 128
            assert args.learning_rate == .0005 and args.weight_decay == args.embedding_weight_decay == .1
            assert task['resources']['gpuCount'] == 8 and task['context']['minRuntime'] == '1h'
            assert args.repetition_filter == ('emo-v1' if key.endswith('-repfilter') else 'none')
            for prefix in ('convert', 'eval'):
                child = state['nodes'][key.replace('train-', prefix + '-', 1)]
                assert child['chain_parent'] == key and child['arguments'][1:] == n['arguments'][1:]
    state['nodes'][prep]['status'] = 'succeeded'
    # A successful scheduler status alone cannot admit the models.
    with pytest.raises(KeyError): campaign.ready_nodes(state)
    monkeypatch.setattr(comparison, 'verify_preparation_receipt', lambda s, n: True)
    assert set(campaign.ready_nodes(state)) == set(auth['runs'])
    key = auth['runs'][0]
    state['nodes'][key]['arguments'] += ['--resume-latest']
    assert key not in campaign.ready_nodes(state)
    with pytest.raises(ValueError, match='already authorized'):
        comparison.authorize_two_runs(state)


def test_dolma2_paths_and_native_validation_are_separate(tmp_path):
    args = submit.parser().parse_args(['train-model', '--root', str(tmp_path), '--dataset', 'dclm',
        '--data-size', '10GB', '--vocab-size', '100278', '--tokenizer-source', 'dolma2'])
    pipeline.validate_args(args)
    assert pipeline.tokenized_dir(args).name == 'dolma2_10GB_100278'
    assert pipeline.tokenized_dir(args, 32000).name == 'default_10GB_32000'
    assert pipeline.tokenizer_file(args) == comparison.tokenizer_path(tmp_path)
    tok, target, extra = validation_data.tokenizer_stream_paths(tmp_path, '10GB', 100278, 'dclm', 'dolma2')
    assert tok == comparison.tokenizer_path(tmp_path)
    assert target.name == 'dolma2_10GB_100278'
    assert extra['tokenizer_source'] == comparison.source_manifest()
    args.dataset = 'dolma'
    with pytest.raises(ValueError, match='limited'): pipeline.validate_args(args)
    with pytest.raises(ValueError, match='mismatch'):
        validation_data.tokenizer_stream_paths(tmp_path, '5GB', 100278, 'dclm', 'dolma2')


def test_dolma2_training_plan_identical_in_chain_and_preserves_eos(tmp_path, monkeypatch):
    from test_dclm_training import test_dclm_training_plan_checks_real_streams_and_separates_output
    # Reuse the real, hash-checked uint32 pool fixture from the DCLM tests.
    test_dclm_training_plan_checks_real_streams_and_separates_output(tmp_path, monkeypatch)
    reference = tmp_path / 'tokenized_data_dclm/default_10GB_32000'
    import shutil
    target = comparison.stream_path(tmp_path)
    shutil.copytree(reference, target)
    meta = pipeline.read_json(target / 'meta.json')
    meta.update(vocab_size=100278, eos_token_id=comparison.EOS, pad_token_id=comparison.PAD, pad_token='<|pad|>')
    pipeline.write_json(target / 'meta.json', meta)
    source = comparison.source_manifest()
    source['files']['tokenizer.json'] = {'bytes': (target/'tokenizer.json').stat().st_size,
                                       'sha256': pipeline.digest(target/'tokenizer.json')}
    monkeypatch.setattr(comparison, 'source_manifest', lambda: source)
    monkeypatch.setattr(comparison, 'verify_tokenizer', lambda root: {})
    recipe = comparison.training_recipe(tmp_path)
    repack = pipeline.read_json(target / 'repack.json')
    pipeline.write_json(target / 'pipeline.json', {'recipe': recipe, 'meta': meta, 'repack': repack})
    monkeypatch.setattr(validation_data, 'validation_metadata', lambda *a, **kw: {'source': kw['tokenizer_source']})
    args = submit.parser().parse_args(['train-model','--root',str(tmp_path),'--dataset','dclm',
        '--data-size','10GB','--vocab-size','100278','--tokenizer-source','dolma2',
        '--learning-rate','.0005','--weight-decay','.1','--embedding-weight-decay','.1'])
    cfg, out = pipeline.training_plan(args)
    assert cfg['pipeline']['tokenizer_source'] == source
    assert cfg['seq_length'] == 4096 and out.parent.name == 'dolma2_10GB_100278'
    for stage in ('convert-model', 'evaluate-model'):
        args.stage = stage
        assert pipeline.training_plan(args) == (cfg, out)
    args.repetition_filter = 'emo-v1'
    filtered, filtered_out = pipeline.training_plan(args)
    assert filtered_out != out and filtered['training']['repetition_filter']
    meta['eos_token_id'] = 0
    pipeline.write_json(target / 'meta.json', meta)
    with pytest.raises(ValueError, match='mismatch'):
        pipeline.training_plan(args)


def test_measured_cpu_progress_and_unknown_queue_time():
    def line(message): return {'timestamp': '2026-10-17T00:00:00Z', 'message': message}
    progress = campaign.log_progress([
        line('DOLMA2_PREPARATION {"total_documents": 1000}'),
        line('TOKENIZATION_PROGRESS {"time": 100, "documents": 100}'),
        line('TOKENIZATION_PROGRESS {"time": 160, "documents": 200}')], comparison.STAGE)
    assert progress['tokenization_eta_seconds'] == 480
    assert 'hashing/validation extra' in progress['eta']


@pytest.mark.parametrize('corruption', ['worker', 'source', 'eos', 'split', 'validation'])
def test_preparation_receipt_checks_worker_sources_and_shared_documents(corruption):
    state = fixture()
    auth = comparison.authorize_two_runs(state)
    node = state['nodes'][auth['preparation_node']]
    source = comparison.source_manifest()
    meta = copy.deepcopy(state['nodes']['dclm-tokens-10gb-32000']['summary']['meta'])
    meta.update(vocab_size=comparison.VOCAB, eos_token_id=comparison.EOS, pad_token_id=comparison.PAD)
    node.update(status='succeeded', receipt={'revision': 'new-worker'}, summary={
        'stage': comparison.STAGE, 'revision': 'new-worker', 'meta': meta,
        'tokenizer': {'source': source, 'vocab_size': comparison.VOCAB,
            'eos_token_id': comparison.EOS, 'pad_token_id': comparison.PAD,
            'sha256': source['files']['tokenizer.json']['sha256']},
        'validation': {'documents': 100, 'bytes': 1000000, 'recipe': {
            'documents_sha256': 'documents-hash', 'tokenizer_source': source,
            'tokenizer_sha256': source['files']['tokenizer.json']['sha256'],
            'vocab_size': comparison.VOCAB, 'validation_id': 'dolma100m-v1',
            'pool': '10GB', 'tokenizer_dataset': 'dclm'}}})
    assert comparison.verify_preparation_receipt(state, node)
    assert set(campaign.ready_nodes(state)) == set(auth['runs'])
    if corruption == 'worker': node['summary']['revision'] = 'wrong'
    elif corruption == 'source': state['nodes']['dclm-tokens-10gb-32000']['receipt']['revision'] = 'wrong'
    elif corruption == 'eos': meta['eos_token_id'] = 0
    elif corruption == 'split': meta['train']['documents'] += 1
    else: node['summary']['validation']['recipe']['documents_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='mismatch'):
        comparison.verify_preparation_receipt(state, node)
