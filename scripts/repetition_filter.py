"""Opt-in training sequence masking using EMO's vendored InstanceFilterConfig.

These are the settings used by the reference EMO run q4q7vy62. Detection,
collation, label masking, loss normalization and metrics stay in EMO itself.
An absent setting means unfiltered, preserving existing run fingerprints.
"""
import copy

FILTER_NAME = 'emo-v1'
FILTER_CHOICES = ('none', FILTER_NAME)
FILTER_SETTINGS = {
    'name': FILTER_NAME,
    'repetition_min_period': 1,
    'repetition_max_period': 13,
    'repetition_max_count': 32,
}


def configure_repetition_filter(config, name=None):
    """Resolve CLI/config consistently; never silently change a saved policy."""
    training = config['training']
    recorded = training.get('repetition_filter')
    if recorded is not None and recorded != FILTER_SETTINGS:
        raise ValueError('Repetition filter differs from the pinned emo-v1 settings')
    name = name if name is not None else FILTER_NAME if recorded else 'none'
    if name not in FILTER_CHOICES:
        raise ValueError(f'Unknown repetition filter: {name}')
    if name == 'none':
        if recorded is not None:
            raise ValueError('Cannot disable the configured repetition filter; start a fresh run')
    else:
        training['repetition_filter'] = copy.deepcopy(FILTER_SETTINGS)
    return name


def build_instance_filter(name='none'):
    if name not in FILTER_CHOICES:
        raise ValueError(f'Unknown repetition filter: {name}')
    if name == 'none':
        return None
    from olmo_core.data.numpy_dataset import InstanceFilterConfig
    return InstanceFilterConfig(**{k: v for k, v in FILTER_SETTINGS.items() if k != 'name'})


def verify_instance_filter(dataset, config):
    """Verify the actual training dataset, including the later-epoch repack path."""
    name = configure_repetition_filter(config)
    actual = dataset.instance_filter_config
    if actual != build_instance_filter(name):
        raise RuntimeError('Training dataset repetition filter differs from the requested configuration')
    return {
        'name': name,
        'dataset_class': type(dataset).__name__,
        'instance_filter_config': actual.as_config_dict() if actual is not None else None,
        'scope': 'training_only',
        'action': 'mask_entire_instance_loss' if actual is not None else 'none',
        'preserves_sequence_order_and_epoch_length': True,
    }
