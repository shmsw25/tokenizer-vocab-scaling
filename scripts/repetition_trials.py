"""Authorize the three explicitly requested DCLM repetition-filter comparisons."""
import argparse
import copy
import fcntl
import hashlib
import json

from campaign import now, prepare_chains
from emo_trials import digest
from loss_curve_health import invalid_result
from pipeline import REPO, read_json, write_json
from repetition_filter import FILTER_NAME, FILTER_SETTINGS
from runtime_batch import GPU_CLUSTERS
from training_recipe import EMO_RECIPE, node_recipe

COHORT = 'dclm_repetition_filter_10gb'
AUDIT = REPO / '.build/dclm-repetition-filter-20261016'


def authorize_three_runs(state):
    """Copy only frozen scientific arguments, never model state or prior receipts."""
    study, nodes = state['study'], state['nodes']
    if study.get(COHORT):
        raise ValueError('Repetition-filter trials already authorized; do not duplicate them')
    additions, roots, required, sources, comparisons = {}, [], [], {}, {}
    original_cohort = study['emo_small_10gb']
    for vocab in (32000, 64000, 128000):
        original_key = f'train-dclm-small-10gb-{vocab}-lr0p0005-emo-dense-v1'
        original = nodes[original_key]
        if (original_key not in original_cohort['runs'] or original['status'] != 'succeeded'
                or original.get('hold_reason') or invalid_result(original)
                or node_recipe(original) != EMO_RECIPE or original.get('dataset') != 'dclm'
                or original['summary']['revision'] != original['receipt']['revision']):
            raise ValueError(f'Unverified unfiltered comparison: {original_key}')
        args = original['arguments']
        expected = {'--dataset': 'dclm', '--data-size': '10GB', '--vocab-size': str(vocab),
                    '--model-size': 'small', '--learning-rate': '0.0005', '--weight-decay': '0.1',
                    '--embedding-weight-decay': '0.1', '--batch-size': '128', '--epochs': '1',
                    '--epoch-fraction': '1', '--seed': '42', '--validation-id': 'dolma100m-v1'}
        if (any(flag not in args or args[args.index(flag)+1] != value for flag, value in expected.items())
                or any(flag in args for flag in ('--repetition-filter', '--resume-latest', '--branch-from'))
                or original['summary']['config']['training'].get('repetition_filter')):
            raise ValueError(f'Unfiltered comparison recipe mismatch: {original_key}')
        for source_key in original['dependencies']:
            source = nodes[source_key]
            frozen = original_cohort['sources'][source_key]
            if (source['status'] != 'succeeded' or source.get('monitor_error')
                    or source['receipt'] != frozen['receipt']
                    or digest(source['summary']) != frozen['summary_sha256']
                    or source['summary']['revision'] != source['receipt']['revision']):
                raise ValueError(f'Frozen training/validation source mismatch: {source_key}')
            sources[source_key] = copy.deepcopy(frozen)
        comparisons[str(vocab)] = {'unfiltered_run': original_key,
            'receipt': copy.deepcopy(original['receipt']), 'summary_sha256': digest(original['summary'])}
        common = [*args[1:], '--repetition-filter', FILTER_NAME]
        suffix = original_key.removeprefix('train-') + '-repfilter'
        dependencies = list(original['dependencies'])
        for prefix, stage in [('train', 'train-model'), ('convert', 'convert-model'), ('eval', 'evaluate-model')]:
            key = f'{prefix}-{suffix}'
            if key in nodes:
                raise ValueError(f'Repetition-filter node already exists: {key}')
            additions[key] = dict(stage=stage, dataset='dclm', arguments=[stage, *common],
                status='pending', dependencies=dependencies, data_size='10GB', vocab_size=vocab,
                model_size='small', epoch_fraction=1., expected_gpus=8 if prefix == 'train' else int(prefix == 'eval'),
                min_runtime='1h', priority='urgent', study_role='explicit_repetition_filter_comparison',
                placement_clusters=list(GPU_CLUSTERS) if prefix == 'train' else ['ai2/jupiter'],
                validation_stream_node=original['validation_stream_node'], comparison_run=original_key)
            if prefix == 'train':
                additions[key].update(chain_evaluation=True, wandb_secret='SEWONM_WANDB_API_KEY',
                    planned_tokens=copy.deepcopy(original['planned_tokens']),
                    validation_provenance=copy.deepcopy(original['validation_provenance']))
                roots.append(key)
            if prefix == 'eval':
                required.append(key)
            dependencies = [key]
    authorized_at = now()
    nodes.update(additions)
    for key in roots:
        study.setdefault('training_submission_exceptions', {})[key] = dict(
            source='explicit_user_request', authorized_at=authorized_at,
            arguments_sha256=hashlib.sha256(json.dumps(nodes[key]['arguments'], separators=(',', ':')).encode()).hexdigest(),
            reason='Three fresh DCLM small/10GB 32K/64K/128K EMO comparisons, changing only repetition-filter to emo-v1')
    study[COHORT] = dict(authorized_at=authorized_at, runs=roots, required=required,
        sources=sources, comparisons=comparisons, repetition_filter=copy.deepcopy(FILTER_SETTINGS),
        training_recipe=EMO_RECIPE, min_runtime='1h',
        scope='Exactly three fresh runs with chained evaluations; no LR descent or automatic replacements')
    study['required'] = sorted(set(study['required']) | set(required))
    prepare_chains(state)
    return study[COHORT]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-three-runs', action='store_true', required=True)
    parser.parse_args()
    path = REPO / '.build/campaign.json'
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (AUDIT / 'state-before.json').exists():
            raise ValueError('Authorization backup exists; inspect rather than rerun')
        state = read_json(path)
        original = copy.deepcopy(state)
        authorization = authorize_three_runs(state)
        if any(state['nodes'][k] != n for k, n in original['nodes'].items()):
            raise ValueError('Existing nodes changed during authorization')
        write_json(AUDIT / 'state-before.json', original)
        write_json(AUDIT / 'authorization.json', authorization)
        write_json(path, state)
        print(json.dumps(authorization, indent=2))


if __name__ == '__main__':
    main()
