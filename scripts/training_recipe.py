"""Version the dense optimization recipe independently of experiment LR/WD/data.

This module has no torch dependency: submission, monitoring, and CPU stages use it.
Unversioned historical campaign nodes keep their original recipe. Fresh CLI runs
use the EMO recipe, and its complete description participates in the run hash.
"""
import copy
import json
from pathlib import Path

EMO_RECIPE = 'emo-dense-v1'
LEGACY_RECIPE = 'legacy-adamw-v1'
RECIPES = (EMO_RECIPE, LEGACY_RECIPE)
EMO_SETTINGS = {
    'name': EMO_RECIPE,
    'reference_commit': '6ef92fc3a45896a8d734ea4f852a3d35950e4abb',
    'optimizer': {'name': 'skip_step_adamw', 'betas': [0.9, 0.95], 'eps': 1e-8,
                  'foreach': True, 'step_increment_bugfix': True,
                  'rolling_interval_length': 128, 'sigma_factor': 6},
    'outlier_policy': 'skip_and_continue',
    'z_loss_multiplier': 1e-5,
    'data_parallel': {'name': 'fsdp', 'param_dtype': 'bfloat16',
                      'reduce_dtype': 'float32', 'wrapping_strategy': 'full'},
    'autocast_precision': None,
    'float32_matmul_precision': 'highest',
    'init_tied_output': True,
}


def configure_recipe(config, name=EMO_RECIPE):
    """Stamp the actual recipe before hashing, device planning, or model building."""
    if name not in RECIPES:
        raise ValueError(f'Unknown training recipe: {name}')
    recorded = config.get('training_recipe')
    if name == LEGACY_RECIPE:
        if recorded is not None or config['model'].get('init_tied_output'):
            raise ValueError('Cannot reinterpret an EMO configuration as the legacy recipe')
        return config  # Preserve every old fingerprint, including absent fields.
    if recorded is not None and recorded != EMO_SETTINGS:
        raise ValueError('Training recipe differs from the pinned EMO recipe')
    # These used to be silently ignored by train.py. Reject contradictory values.
    expected = {'betas': [0.9, 0.95], 'eps': 1e-8, 'z_loss_multiplier': 1e-5,
                'autocast_precision': None, 'float32_matmul_precision': 'highest'}
    for key, value in expected.items():
        if key in config['training'] and config['training'][key] != value:
            raise ValueError(f'training.{key} conflicts with {EMO_RECIPE}')
    if config['model'].get('init_tied_output', True) is not True:
        raise ValueError('EMO requires output-head initialization of tied weights')
    if 'health_check' in config['training']:
        # This legacy switch installs the study's extra stop rules, not EMO's
        # optimizer handling. Record it as disabled rather than saving a true
        # flag for a callback that the new recipe will never install.
        config['training']['health_check'] = False
    config['training_recipe'] = copy.deepcopy(EMO_SETTINGS)
    config['model']['init_tied_output'] = True
    return config


def checkpoint_recipe_guard(checkpoint, config):
    """Never restore AdamW/DDP state into SkipStepAdamW/FSDP (or vice versa)."""
    path = Path(checkpoint) / 'config.json'
    if not path.is_file():
        raise ValueError(f'Checkpoint lacks its saved configuration: {path}')
    saved = json.loads(path.read_text())
    if saved.get('training_recipe') != config.get('training_recipe'):
        raise ValueError('Checkpoint training recipe mismatch; start a fresh run')
    if saved.get('training', {}).get('repetition_filter') != config.get('training', {}).get('repetition_filter'):
        raise ValueError('Checkpoint repetition filter mismatch; start a fresh run')


def node_recipe(node):
    """Only an explicitly versioned node opts into new health semantics."""
    arguments = node.get('arguments', [])
    for i, value in enumerate(arguments):
        if value.startswith('--training-recipe='):
            return value.split('=', 1)[1]
        if value == '--training-recipe' and i + 1 < len(arguments):
            return arguments[i + 1]
    return LEGACY_RECIPE


def campaign_arguments(node):
    """Pin old campaign commands without mutating receipts or authorization hashes."""
    arguments = list(node['arguments'])
    if node['stage'] in ('train-model', 'convert-model', 'evaluate-model'):
        if not any(a == '--training-recipe' or a.startswith('--training-recipe=') for a in arguments):
            arguments += ['--training-recipe', LEGACY_RECIPE]
    return arguments
