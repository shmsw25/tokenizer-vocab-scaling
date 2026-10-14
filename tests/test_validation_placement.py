"""Standalone rescoring may use spare Weka GPUs without changing training placement."""
from types import SimpleNamespace

import pytest
from google.protobuf import json_format

import adaptive_study
import submit
from test_adaptive_study import fixture


def args_for(stage, *extra):
    base = [stage, '--vocab-size', '32000']
    if stage == 'validate-model':
        base += ['--model-directory', '/oe-training-default/example']
    return submit.parser().parse_args(base + list(extra))


def test_validation_spec_preserves_one_gpu_weka_and_runtime():
    args = args_for('validate-model')
    submit.validate_placement(args)
    task = submit.parse_spec(submit.make_spec(args)).to_json()['tasks'][0]
    assert task['constraints'] == {'cluster': ['ai2/jupiter', 'ai2/saturn', 'ai2/neptune']}
    assert task['resources']['gpuCount'] == 1
    assert task['context']['minRuntime'] == 1_800_000_000_000
    assert task['datasets'][1]['source'] == {'weka': 'oe-training-default'}
    submit.validate_placement(args_for('train-model', '--cluster', 'ai2/jupiter', 'ai2/saturn'))
    for stage in ('evaluate-model',):
        with pytest.raises(ValueError, match='exactly one'):
            submit.validate_placement(args_for(stage, '--cluster', 'ai2/jupiter', 'ai2/saturn'))


def test_every_candidate_is_checked_and_legacy_postprocessing_stays_h100(monkeypatch):
    types = {'jupiter': 'GPU_TYPE_NVIDIA_H100', 'saturn': 'GPU_TYPE_NVIDIA_A100_80GB',
             'neptune': 'GPU_TYPE_NVIDIA_L40S'}
    seen = []
    def nodes(cluster):
        seen.append(cluster.name)
        return [SimpleNamespace(kind=types[cluster.name], node_resources=SimpleNamespace(gpu_ids=list(range(8))))]
    monkeypatch.setattr(json_format, 'MessageToDict', lambda node: {'nodeResources': {'gpuType': node.kind}})
    beaker = SimpleNamespace(node=SimpleNamespace(list=nodes))
    clusters = [SimpleNamespace(name=name) for name in types]
    submit.verify_gpu_clusters(beaker, args_for('validate-model'), clusters)
    assert seen == ['jupiter', 'saturn', 'neptune']
    with pytest.raises(RuntimeError, match='saturn'):
        submit.verify_gpu_clusters(beaker, args_for('evaluate-model'), clusters)
    submit.verify_gpu_clusters(beaker, args_for('train-model'), clusters)
    types['neptune'] = 'GPU_TYPE_NVIDIA_V100'
    with pytest.raises(RuntimeError, match='neptune'):
        submit.verify_gpu_clusters(beaker, args_for('validate-model'), clusters)


def test_controller_places_new_validation_without_rewriting_existing_receipts():
    state = fixture()
    state['study'].update(validation_node='prepare-validation',
                          validation_placement_clusters=['ai2/saturn', 'ai2/neptune'])
    key = 'train-small-5gb-32000'
    state['nodes'][key]['summary']['model_dir'] = '/oe-training-default/example'
    state['nodes'][adaptive_study.eval_key(key)].update(status='succeeded', summary={})
    adaptive_study.sync_shared_validation(state)
    node = state['nodes']['validation-small-5gb-32000']
    assert node['placement_clusters'] == ['ai2/saturn', 'ai2/neptune']
    assert node['min_runtime'] == '30m'
    node.update(status='queued', receipt={'workload_id': 'existing'})
    state['study']['validation_placement_clusters'] = list(submit.VALIDATION_CLUSTERS)
    adaptive_study.sync_shared_validation(state)
    assert node['receipt'] == {'workload_id': 'existing'}
    assert node['placement_clusters'] == ['ai2/saturn', 'ai2/neptune']
