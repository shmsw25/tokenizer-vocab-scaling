"""Offline regression and integration checks; no Beaker jobs or full download."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pipeline
import submit
from beaker import BeakerExperimentSpec


def args_for(stage, *extra):
    return submit.parser().parse_args([stage, *extra])


@pytest.mark.parametrize('stage', pipeline.STAGES)
def test_beaker_resources_and_schema(stage):
    extra = [] if stage == 'download-data' else ['--vocab-size', '48000']
    args = args_for(stage, *extra)
    pipeline.validate_args(args)
    spec = submit.make_spec(args)
    parsed = BeakerExperimentSpec.from_json(copy.deepcopy(spec)).to_json()
    task = parsed['tasks'][0]
    assert task['resources']['gpuCount'] == (1 if stage == 'train-model' else 0)
    assert task['constraints']['cluster'] == ['ai2/jupiter']
    assert task['datasets'][1] == {
        'mountPath': '/oe-training-default', 'source': {'weka': 'oe-training-default'}}
    assert task['arguments'][:3] == [stage, '--root', pipeline.DEFAULT_ROOT]
    if stage != 'download-data':
        assert task['arguments'][3:7] == ['--vocab-size', '48000', '--data-size', '100GB']
    assert '--seq-length' not in task['arguments']


def test_invalid_gpu_and_vocab_arguments():
    with pytest.raises(ValueError, match='divisible'):
        pipeline.validate_args(args_for('train-model', '--vocab-size', '48000', '--gpus', '3'))
    with pytest.raises(ValueError, match='at least 257'):
        pipeline.validate_args(args_for('train-tokenizer', '--vocab-size', '256'))


def make_meta_fixture(tmp_path, vocab, ratio, bytes_per_split):
    args = args_for('train-model', '--root', str(tmp_path), '--vocab-size', str(vocab))
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
    make_meta_fixture(tmp_path, 32000, 0.5, 2_000_000)
    args = make_meta_fixture(tmp_path, 48000, 0.4, 2_000_000)
    cfg, output = pipeline.training_plan(args)
    assert cfg['seq_length'] == round(4096 * 0.4 / 0.5)
    assert cfg['pipeline']['scheduler'] == 'WSD'
    assert cfg['training']['gradient_accumulation_steps'] == 128
    assert output.parent == tmp_path / 'models/small/default_100GB_48000'
    assert 'lr0.0003_wd0.1_bs128_seq3277_e1_seed42' in output.name
    args.learning_rate = 0.002
    assert pipeline.training_plan(args)[1] != output
    args.gpus = 2
    cfg, _ = pipeline.training_plan(args)
    assert cfg['training']['gradient_accumulation_steps'] == 64
    meta_path = pipeline.tokenized_dir(args, 32000) / 'meta.json'
    meta = pipeline.read_json(meta_path)
    meta['train']['bytes'] += 1
    pipeline.write_json(meta_path, meta)
    with pytest.raises(ValueError, match='train.bytes'):
        pipeline.training_plan(args)


def test_missing_reference_fails(tmp_path):
    args = make_meta_fixture(tmp_path, 48000, 0.4, 2_000_000)
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
    scheduler = WSD(warmup_fraction=0.03, decay_fraction=0.1, decay_min_lr=0.0)
    assert scheduler.get_lr(0.001, 15, 1000) == pytest.approx(0.0005)
    assert scheduler.get_lr(0.001, 500, 1000) == pytest.approx(0.001)
    assert scheduler.get_lr(0.001, 950, 1000) == pytest.approx(0.0005)
    assert scheduler.get_lr(0.001, 1000, 1000) == pytest.approx(0)
    model = TransformerConfig.llama_like(d_model=64, n_layers=1, n_heads=4, vocab_size=512,
                                        tie_word_embeddings=True,
                                        attn_backend=AttentionBackendName.torch).build(init_device='cpu')
    assert model.embeddings.weight is model.lm_head.w_out.weight
