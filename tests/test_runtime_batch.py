import copy
import json
from pathlib import Path

import pytest

import pipeline
import campaign
import runtime_batch
import submit
from test_pipeline import make_meta_fixture


def device(family):
    return {'name': f'NVIDIA {family}', 'total_memory': (48 if family == 'L40S' else 80) * 1024**3,
            'compute_capability': [8, 9] if family == 'L40S' else [9, 0] if family == 'H100' else [8, 0]}


@pytest.mark.parametrize('model', ['small', 'medium', 'medium_large'])
@pytest.mark.parametrize('vocab', [32000, 64000, 128000])
@pytest.mark.parametrize('family', ['H100', 'A100', 'L40S'])
def test_runtime_policy_preserves_batch_and_scientific_config(model, vocab, family):
    cfg = json.loads((pipeline.REPO / 'configs' / f'{model}.json').read_text())
    micro = 4 if model == 'medium_large' and vocab == 128000 else cfg['training']['micro_batch_size']
    cfg['training'].update(micro_batch_size=micro, gradient_accumulation_steps=128 // (8 * micro))
    cfg['pipeline'] = {'gpus': 8, 'global_batch_size': 128}
    original = copy.deepcopy(cfg)
    plan = runtime_batch.make_plan(cfg, model, vocab, [device(family)] * 8)
    assert cfg == original
    expected = micro if family != 'L40S' else {'small': 8, 'medium': 4, 'medium_large': 1 if vocab == 128000 else 2}[model]
    assert plan['micro_batch_size'] == expected
    for rank in range(8):
        effective = copy.deepcopy(cfg)
        runtime_batch.apply_plan(effective, plan, 8, rank, device(family))
        assert effective['training']['micro_batch_size'] * 8 * effective['training']['gradient_accumulation_steps'] == 128
        effective.pop('runtime_batch')
        for field in ('micro_batch_size', 'gradient_accumulation_steps'):
            effective['training'][field] = original['training'][field]
        assert effective == original


def test_invalid_assignments_and_changed_recipe_fail_closed(tmp_path):
    args = make_meta_fixture(tmp_path, 32000, .5, 4_000_000)
    cfg, output = pipeline.training_plan(args)
    plan = runtime_batch.make_plan(cfg, 'small', 32000, [device('L40S')] * 8)
    for devices in ([device('L40S')] * 7, [device('L40S')] * 7 + [device('H100')],
                    [dict(device('A100'), total_memory=40 * 1024**3)] * 8):
        with pytest.raises(ValueError):
            runtime_batch.make_plan(cfg, 'small', 32000, devices)
    changed = copy.deepcopy(cfg)
    changed['training']['learning_rate'] /= 2
    with pytest.raises(ValueError, match='configuration'):
        runtime_batch.apply_plan(changed, plan, 8, 0, device('L40S'))
    with pytest.raises(ValueError, match='Assigned GPU'):
        runtime_batch.apply_plan(copy.deepcopy(cfg), plan, 8, 0, device('H100'))
    corrupt = dict(plan, gradient_accumulation_steps=1)
    with pytest.raises(ValueError, match='configuration'):
        runtime_batch.apply_plan(copy.deepcopy(cfg), corrupt, 8, 0, device('L40S'))
    assert pipeline.training_plan(args) == (cfg, output)


def test_pipeline_runs_hardware_preflight_before_torchrun_without_changing_fingerprint(tmp_path, monkeypatch):
    args = make_meta_fixture(tmp_path, 32000, .5, 4_000_000)
    config, output = pipeline.training_plan(args)
    calls = []
    def run(command):
        calls.append(list(map(str, command)))
        if 'torch.distributed.run' in command:
            assert len(calls) == 2 and calls[0][1].endswith('/scripts/runtime_batch.py')
            assert command[command.index('--config') + 1] == output / 'launch_config.json'
            assert '--runtime-batch-plan' in command
            pipeline.write_json(output / 'training_metrics.json', {'steps': 1})
    monkeypatch.setattr(pipeline, 'run', run)
    monkeypatch.setattr(pipeline, 'latest_checkpoint', lambda _: output / 'step1')
    pipeline.train_model(args)
    assert pipeline.read_json(output / 'launch_config.json') == config
    assert pipeline.read_json(output / 'training_summary.json')['config'] == config
    assert pipeline.training_plan(args) == (config, output)


def test_training_spec_uses_eight_gpus_on_any_supported_cluster():
    args = submit.parser().parse_args(['train-model', '--vocab-size', '128000', '--chain-evaluation'])
    submit.validate_args(args)
    submit.validate_placement(args)
    task = submit.make_spec(args)['tasks'][0]
    assert task['constraints'] == {'cluster': list(runtime_batch.GPU_CLUSTERS)}
    assert task['resources']['gpuCount'] == 8
    assert task['context']['minRuntime'] == '3h'
    assert {'name': 'TOKENIZER_CHAIN_EVALUATION', 'value': '1'} in task['envVars']


def test_later_explicit_retry_uses_new_placement_and_archives_runtime_evidence():
    node = {'stage': 'train-model', 'status': 'failed', 'dependencies': [], 'arguments': ['train-model'],
            'placement_clusters': ['ai2/jupiter'], 'receipt': {'workload_id': 'old'},
            'runtime_batch': {'micro_batch_size': 16}, 'runtime_batch_verified': {'step': 1}}
    state = {'campaign': 'test', 'study': {'training_placement_clusters': list(runtime_batch.GPU_CLUSTERS)},
             'nodes': {'train': node}}
    campaign.retry_node(state, 'train')
    assert node['placement_clusters'] == list(runtime_batch.GPU_CLUSTERS)
    assert 'runtime_batch' not in node and 'runtime_batch_verified' not in node
    assert node['previous_attempts'][-1]['runtime_batch'] == {'micro_batch_size': 16}
    assert node['previous_attempts'][-1]['placement_clusters'] == ['ai2/jupiter']
    assert node['arguments'] == ['train-model']
