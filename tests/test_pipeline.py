"""Offline regression and integration checks; no Beaker jobs or full download."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pipeline
import submit
import campaign


def args_for(stage, *extra):
    return submit.parser().parse_args([stage, *extra])


@pytest.mark.parametrize('stage', pipeline.STAGES)
def test_beaker_resources_and_schema(stage):
    extra = ([] if stage == 'download-data' else ['--data-size', '2.5GB']
             if stage == 'sample-pool' else ['--vocab-size', '48000'])
    args = args_for(stage, *extra)
    pipeline.validate_args(args)
    spec = submit.make_spec(args)
    parsed = submit.parse_spec(spec).to_json()
    task = parsed['tasks'][0]
    expected_gpus = 8 if stage == 'train-model' else int(stage == 'evaluate-model')
    assert task['resources']['gpuCount'] == expected_gpus
    if stage == 'train-model':
        assert task['context']['minRuntime'] == '0s'
        assert 'preemptible' not in task['context']
    if stage in pipeline.RAW_STAGES:
        assert args.cluster is None
        assert task['constraints']['cluster'] == [submit.WEKA_CLUSTERS_PLACEHOLDER]
        resolved = submit.make_spec(args, resolved_clusters=['ai2/ceres', 'ai2/phobos'])
        assert resolved['tasks'][0]['constraints']['cluster'] == ['ai2/ceres', 'ai2/phobos']
        assert resolved['tasks'][0]['resources']['gpuCount'] == 0
    else:
        assert task['constraints']['cluster'] == ['ai2/jupiter' if expected_gpus else 'ai2/phobos']
    assert task['datasets'][1] == {
        'mountPath': '/oe-training-default', 'source': {'weka': 'oe-training-default'}}
    assert task['arguments'][:3] == [stage, '--root', pipeline.DEFAULT_ROOT]
    if stage == 'sample-pool':
        assert task['arguments'][3:] == ['--data-size', '2.5GB', '--sampling-seed', '42']
    elif stage != 'download-data':
        assert task['arguments'][3:7] == ['--vocab-size', '48000', '--data-size', '100GB']
    assert '--seq-length' not in task['arguments']


def test_invalid_gpu_and_vocab_arguments():
    with pytest.raises(ValueError, match='divisible'):
        pipeline.validate_args(args_for('train-model', '--vocab-size', '48000', '--gpus', '3'))
    with pytest.raises(ValueError, match='at least 257'):
        pipeline.validate_args(args_for('train-tokenizer', '--vocab-size', '256'))


def test_download_excludes_only_explicit_allocation_rejections(tmp_path):
    from types import SimpleNamespace
    from beaker.exceptions import BeakerError, BeakerPermissionsError
    args = args_for('download-data', '--spec-file', str(tmp_path / 'spec.json'))
    spec = submit.make_spec(args, resolved_clusters=['ai2/holmes', 'ai2/triton', 'ai2/phobos', 'ai2/jupiter'])
    calls = []
    def create(**kwargs):
        clusters = kwargs['spec'].to_json()['tasks'][0]['constraints']['cluster']
        calls.append(clusters[:])
        if 'ai2/holmes' in clusters:
            raise BeakerError('[code=422] task "main": cluster "ai2/holmes" requires an allocation '
                              'for workloads with a minimum runtime, and workspace lacks one')
        if 'ai2/triton' in clusters:
            raise BeakerPermissionsError('task "main": cluster "ai2/triton" doesn\'t allow tasks')
        return 'created'
    client = SimpleNamespace(experiment=SimpleNamespace(create=create))
    assert submit.create_experiment(client, args, None, 'test', spec) == 'created'
    assert calls == [['ai2/holmes', 'ai2/triton', 'ai2/phobos', 'ai2/jupiter'],
                     ['ai2/triton', 'ai2/phobos', 'ai2/jupiter'], ['ai2/phobos', 'ai2/jupiter']]
    assert json.loads(args.spec_file.read_text())['tasks'][0]['resources']['gpuCount'] == 0
    def uncertain(**kwargs):
        raise BeakerError('[code=503] unknown submission outcome')
    client.experiment.create = uncertain
    with pytest.raises(BeakerError, match='unknown submission outcome'):
        submit.create_experiment(client, args, None, 'test', spec)


def make_meta_fixture(tmp_path, vocab, ratio, bytes_per_split, size='100GB'):
    args = args_for('train-model', '--root', str(tmp_path), '--vocab-size', str(vocab), '--data-size', size)
    raw = pipeline.raw_file(args)
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        raw.write_text('{"text": "synthetic fixture"}\n')
    directory = pipeline.tokenized_dir(args)
    directory.mkdir(parents=True)
    (directory / 'tokenizer.json').write_text('{}')
    meta = {'vocab_size': vocab, 'tokens_per_byte': ratio, 'data_files': [str(raw)],
            'seed': 42, 'eval_fraction': 0.01}
    for split in ('train', 'eval'):
        tokens = int(bytes_per_split * ratio)
        meta[split] = {'tokens': tokens, 'bytes': bytes_per_split, 'documents': 100}
        with (directory / f'{split}.bin').open('wb') as stream:
            stream.truncate(tokens * 4)
    pipeline.write_json(directory / 'meta.json', meta)
    pipeline.write_json(directory / 'pipeline.json', {
        'raw': pipeline.raw_signature(raw), 'tokenizer_sha256': pipeline.digest(directory / 'tokenizer.json')})
    return args


def test_scaled_context_fixed_batch_and_output(tmp_path):
    make_meta_fixture(tmp_path, 32000, 0.5, 4_000_000)
    args = make_meta_fixture(tmp_path, 48000, 0.4, 4_000_000)
    cfg, output = pipeline.training_plan(args)
    assert cfg['seq_length'] == round(4096 * 0.4 / 0.5)
    assert cfg['pipeline']['scheduler'] == 'WSD'
    assert cfg['training']['gradient_accumulation_steps'] == 1
    assert cfg['training']['warmup_steps'] == 96
    assert 'warmup_fraction' not in cfg['training']
    assert '_wu' not in output.name and '_decay' not in output.name and '_e1_' not in output.name
    assert output.parent == tmp_path / 'models/small/default_100GB_48000'
    assert 'lr0.0003_wd0.1_bs256_seq3277_seed42' in output.name
    assert '_wt_EmbedWD_' in output.name
    for stage in ('convert-model', 'evaluate-model'):
        downstream = copy.deepcopy(args)
        downstream.stage = stage
        assert pipeline.training_plan(downstream)[1] == output
    args.weight_decay = 0
    no_wd_path = pipeline.training_plan(args)[1]
    assert '_wt_' in no_wd_path.name and 'EmbedWD' not in no_wd_path.name
    args.weight_decay = None
    args.learning_rate = 0.002
    assert pipeline.training_plan(args)[1] != output
    args.gpus = 2
    cfg, _ = pipeline.training_plan(args)
    assert cfg['training']['gradient_accumulation_steps'] == 4
    meta_path = pipeline.tokenized_dir(args, 32000) / 'meta.json'
    meta = pipeline.read_json(meta_path)
    meta['train']['bytes'] += 1
    pipeline.write_json(meta_path, meta)
    with pytest.raises(ValueError, match='train.bytes'):
        pipeline.training_plan(args)


def test_missing_reference_fails(tmp_path):
    args = make_meta_fixture(tmp_path, 48000, 0.4, 4_000_000)
    with pytest.raises(FileNotFoundError, match='default_100GB_32000'):
        pipeline.training_plan(args)


def test_uniform_sample_and_cpu_tokenizers(tmp_path):
    full = tmp_path / 'raw_text/250GB/dolma3_250gb.jsonl'
    full.parent.mkdir(parents=True)
    records = [{'id': i, 'text': f'Document {i}: café 한국어 test data. ' * 12} for i in range(1000)]
    full.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records))
    args = args_for('download-data', '--root', str(tmp_path))
    pipeline.download_data(args)  # existing tiny 250GB-path fixture skips the network download
    sample = tmp_path / 'raw_text/100GB/dolma3_100gb.jsonl'
    selected = [json.loads(line) for line in sample.read_text().splitlines()]
    assert len(selected) == 400
    assert len({x['id'] for x in selected}) == 400
    assert [x['id'] for x in selected] == sorted(x['id'] for x in selected)
    assert all(row == records[row['id']] for row in selected)
    splits = []
    for vocab in (300, 512):
        args = args_for('train-tokenizer', '--root', str(tmp_path), '--vocab-size', str(vocab))
        pipeline.train_tokenizer(args)
        pipeline.train_tokenizer(args)  # completed work is reused
        pipeline.tokenize_data(args)
        pipeline.tokenize_data(args)
        meta = pipeline.completed_meta(pipeline.tokenized_dir(args), vocab)
        assert meta['train']['documents'] + meta['eval']['documents'] == 400
        splits.append((meta['train']['documents'], meta['eval']['documents'],
                       meta['train']['bytes'], meta['eval']['bytes']))
    assert splits[0] == splits[1]
    # A modified source must not silently reuse tokenized outputs.
    sample.write_text(sample.read_text() + '{"text":"changed"}\n')
    with pytest.raises(ValueError, match='different source'):
        pipeline.tokenize_data(args)


def test_wsd_and_patched_embeddings():
    from olmo_core.optim import WSD
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig
    scheduler = WSD(warmup=96, decay_fraction=0.1, decay_min_lr=0.0)
    assert scheduler.get_lr(0.001, 48, 1000) == pytest.approx(0.0005)
    assert scheduler.get_lr(0.001, 500, 1000) == pytest.approx(0.001)
    assert scheduler.get_lr(0.001, 950, 1000) == pytest.approx(0.0005)
    assert scheduler.get_lr(0.001, 1000, 1000) == pytest.approx(0)
    model = TransformerConfig.llama_like(d_model=64, n_layers=1, n_heads=4, vocab_size=512,
                                        tie_word_embeddings=True,
                                        attn_backend=AttentionBackendName.torch).build(init_device='cpu')
    assert model.embeddings.weight is model.lm_head.w_out.weight
    # Check the actual optimizer group and update, not just a config flag.
    from olmo_core.optim import AdamWConfig
    import torch
    optimizer = AdamWConfig(lr=0.01, weight_decay=0.1, fused=False).build(model)
    embedding = model.embeddings.weight
    groups = [g for g in optimizer.param_groups if any(p is embedding for p in g['params'])]
    assert len(groups) == 1 and groups[0]['weight_decay'] == 0.1
    original = embedding.detach().clone()
    embedding.grad = torch.zeros_like(embedding)
    optimizer.step()
    torch.testing.assert_close(embedding, original * 0.999)
    for size in ('small', 'medium', 'medium_large'):
        cfg = pipeline.read_json(pipeline.REPO / 'configs' / f'{size}.json')
        assert cfg['model']['tie_word_embeddings'] is True
        configured = TransformerConfig.llama_like(vocab_size=32000, **cfg['model']).build(init_device='meta')
        assert configured.embeddings.weight is configured.lm_head.w_out.weight


def test_campaign_dependencies_retry_and_report(tmp_path):
    state = {'campaign': 'test', 'pool_sizes': list(pipeline.POOL_SIZES),
             'training_pools': list(pipeline.TRAINING_POOLS),
             'nodes': campaign.make_nodes([32000, 64000, 128000], ['small', 'medium', 'medium_large'])}
    nodes = state['nodes']
    assert len(nodes) == 115
    assert len([n for n in nodes.values() if n['stage'] == 'train-model']) == 27
    assert len([n for n in nodes.values() if n['stage'] == 'train-tokenizer']) == 18
    assert len([n for n in nodes.values() if n['stage'] == 'tokenize-data']) == 9
    assert campaign.ready_nodes(state) == ['download']
    nodes['download']['status'] = 'succeeded'
    sample_keys = {f'sample-{campaign.size_key(s)}' for s in pipeline.POOL_SIZES}
    assert set(campaign.ready_nodes(state)) == sample_keys
    for key in sample_keys - {'sample-80gb'}:
        nodes[key]['status'] = 'succeeded'
    assert not any(key.startswith('bpe-') for key in campaign.ready_nodes(state))
    nodes['sample-80gb']['status'] = 'succeeded'
    assert len(campaign.ready_nodes(state)) == 18
    nodes['bpe-5gb-64000']['status'] = 'succeeded'
    nodes['tokens-5gb-64000']['status'] = 'succeeded'
    assert not any(key.startswith('train-') for key in campaign.ready_nodes(state))
    nodes['tokens-10gb-32000']['status'] = 'succeeded'
    assert 'train-small-5gb-64000' not in campaign.ready_nodes(state)
    nodes['tokens-5gb-32000']['status'] = 'succeeded'
    assert 'train-small-5gb-64000' in campaign.ready_nodes(state)
    nodes['train-small-5gb-64000'].update(status='failed', receipt={'workload_id': 'old'})
    campaign.retry_node(state, 'train-small-5gb-64000')
    assert nodes['train-small-5gb-64000']['previous_attempts'][0]['receipt']['workload_id'] == 'old'
    assert nodes['train-small-5gb-64000']['name'].endswith('-retry1')
    assert 'receipt' not in nodes['train-small-5gb-64000']
    with pytest.raises(ValueError, match='confirmed failed'):
        campaign.retry_node(state, 'train-small-5gb-64000')
    report = tmp_path / 'report.html'
    campaign.render_report(state, report)
    assert not state['complete']
    assert 'In progress' in report.read_text()
    assert 'Dolma BPB' in report.read_text()
    assert '<th>Pool</th>' in report.read_text() and '2.5GB' in report.read_text()


def test_campaign_migration_preserves_parent_and_refuses_active_jobs():
    state = {'nodes': {'download': {'stage': 'download-data', 'status': 'running',
                                   'receipt': {'workload_id': 'keep-me'}, 'previous_attempts': ['old']},
                       'bpe-32000': {'stage': 'train-tokenizer', 'status': 'pending'}},
             'submission_hold': 'Updating plan'}
    parent = copy.deepcopy(state['nodes']['download'])
    submitted = copy.deepcopy(state)
    submitted['nodes']['bpe-32000']['name'] = 'uncertain-outcome'
    with pytest.raises(ValueError, match='submitted legacy'):
        campaign.migrate_pools(submitted, [32000, 64000, 128000], list(pipeline.MODEL_MICROBATCHES),
                               pipeline.POOL_SIZES, pipeline.TRAINING_POOLS)
    campaign.migrate_pools(state, [32000, 64000, 128000], list(pipeline.MODEL_MICROBATCHES),
                           pipeline.POOL_SIZES, pipeline.TRAINING_POOLS)
    assert state['nodes']['download'] == parent
    assert 'bpe-32000' in state['plan_history'][0]['nodes']
    assert 'submission_hold' not in state
    with pytest.raises(ValueError, match='already migrated'):
        campaign.migrate_pools(state, [32000], ['small'], pipeline.POOL_SIZES, pipeline.TRAINING_POOLS)


def test_monitor_eta_uses_observed_progress():
    lines = [{'timestamp': '2026-10-12T19:00:00Z',
              'message': 'Downloaded 1,000 records, 1.000 / 250 GB of text'},
             {'timestamp': '2026-10-12T19:01:00Z',
              'message': 'Downloaded 2,000 records, 2.000 / 250 GB of text'}]
    assert '~4.1 h' in campaign.log_progress(lines, 'download-data')['eta']
    lines.append({'timestamp': '2026-10-12T19:02:00Z',
                  'message': '[step=100/1000,epoch=1,eta=3h20m] CE loss: 3.1'})
    progress = campaign.log_progress(lines, 'train-model')
    assert progress['eta'] == '3h20m' and progress['step'] == 100


def test_sampling_progress_supersedes_completed_download():
    lines = [{'timestamp': '2026-10-13T00:55:02Z',
              'message': 'Downloaded 36,565,000 records, 250.001 / 250 GB of text'},
             {'timestamp': '2026-10-13T00:55:03Z',
              'message': 'Running: python data/repeat_dolma.py --sample-fraction 0.4'}]
    assert 'Sampling ETA not yet' in campaign.log_progress(lines, 'download-data')['eta']
    lines.append({'timestamp': '2026-10-13T00:55:51Z', 'message':
                  '\rCounting records: 14%|##| 46G/327G [00:53<08:34, 547MB/s]'
                  '\rCounting records: 46%|####| 150G/327G [02:54<04:53, 601MB/s]'
                  '\rCounting records: 46%|####| 150G/327G'})
    progress = campaign.log_progress(lines, 'download-data')
    assert progress['text'].startswith('Counting records: 46%')
    assert progress['eta'] == '04:53 for record counting; sample writing follows'
    lines.append({'timestamp': '2026-10-13T01:02:00Z', 'message':
                  '\rSampling records: 20%|##| 2.93M/14.6M [01:30<06:00, 32.5k records/s]'
                  '\rSampling records: 20%|'})
    assert campaign.log_progress(lines, 'download-data')['eta'] == '06:00 for sample writing'
