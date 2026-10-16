"""A Beaker preemption must not leave monitoring attached to the old W&B run."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign import record_training_job_change
from loss_curve_health import history_paths


def example():
    node = {'stage': 'train-model', 'job_id': 'old',
            'receipt': {'workload_id': 'same-workload'},
            'arguments': ['train-model', '--training-recipe', 'emo-dense-v1'],
            'reported_metrics': {'step': 1641, 'wandb_path': 'entity/project/old'},
            'wandb_observation': {'state': 'crashed'},
            'runtime_batch_verified': {'step': 1, 'plan': {'gpu_family': 'H100'}},
            'progress': {'text': '1641/4574'}, 'loss_curve_review': {'status': 'no_spike_detected'},
            'user_hold': True, 'loss_health_decision': {'decision': 'valid'}}
    previous = {'id': 'old', 'workloadId': 'same-workload',
                'status': {'status': 'STATUS_CANCELED',
                           'canceledCode': 'CANCELATION_CODE_SYSTEM_PREEMPTION'}}
    return node, previous


def test_restart_preserves_history_and_allows_lower_step_new_segment():
    node, previous = example()
    before = copy.deepcopy(node)
    record_training_job_change(node, 'new', previous)
    assert node['receipt'] == before['receipt']
    assert node['arguments'] == before['arguments']
    assert node['user_hold'] and node['loss_health_decision'] == before['loss_health_decision']
    assert 'reported_metrics' not in node and 'runtime_batch_verified' not in node
    assert 'progress' not in node and 'loss_curve_review' not in node
    assert node['previous_attempts'][0]['reported_metrics'] == before['reported_metrics']
    assert node['previous_attempts'][0]['job_status'] == previous['status']
    node['reported_metrics'] = {'step': 1, 'wandb_path': 'entity/project/new'}
    assert history_paths(node) == ['entity/project/old', 'entity/project/new']
    record_training_job_change(node, 'new', previous)
    assert len(node['previous_attempts']) == 1
    assert node['reported_metrics']['step'] == 1


@pytest.mark.parametrize('field,value', [('id', 'unrelated'), ('workloadId', 'unrelated'),
    ('status', {'status': 'STATUS_RUNNING'})])
def test_restart_fails_closed_on_wrong_identity_or_live_predecessor(field, value):
    node, previous = example()
    before = copy.deepcopy(node)
    previous[field] = value
    with pytest.raises(ValueError):
        record_training_job_change(node, 'new', previous)
    assert node == before


def test_first_job_does_not_archive_or_clear_observations():
    node, previous = example()
    node.pop('job_id')
    before = copy.deepcopy(node)
    record_training_job_change(node, 'new', previous)
    assert node == before
