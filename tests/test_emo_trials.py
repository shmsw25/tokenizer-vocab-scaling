import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import adaptive_study
import campaign
import submit
from emo_trials import authorize_six_runs
from loss_curve_health import invalid_result
from training_recipe import EMO_RECIPE, campaign_arguments


def fixture():
    state = {'study': {'root': '/oe-training-default/sewonm/tokenizer_project',
        'required': [], 'validation_node': 'prepare-validation',
        'training_submission_hold': {'reason': 'Preserve user hold'}}, 'nodes': {}}
    corpus = {'sha256': 'documents-hash', 'documents': 100, 'text_bytes': 1000000}
    def validation(dataset, vocab):
        return {'documents': 100, 'bytes': 1000000, 'recipe': {
            'documents_sha256': 'documents-hash', 'pool': '10GB', 'vocab_size': vocab,
            'validation_id': 'dolma100m-v1', 'tokenizer_dataset': dataset}}
    def add(key, stage, **summary):
        state['nodes'][key] = dict(stage=stage, status='succeeded', dependencies=[], arguments=[],
            receipt={'workload_id': key, 'revision': 'pinned'},
            summary=dict(stage=stage, revision='pinned', **summary))
    add('prepare-validation', 'prepare-validation', validation=corpus,
        streams={f'default_10GB_{v}': validation('dolma', v) for v in (32000, 64000, 128000)})
    for dataset in ('dolma', 'dclm'):
        for vocab in (32000, 64000, 128000):
            prefix = 'dclm-' if dataset == 'dclm' else ''
            add(f'{prefix}tokens-10gb-{vocab}', 'tokenize-dclm-data' if prefix else 'tokenize-data',
                meta={'vocab_size': vocab, 'seed': 42, 'eval_fraction': .01, 'data_files': [dataset],
                      'tokens_per_byte': .25, 'train': {'tokens': 10000000, 'bytes': 40000000, 'documents': 1000},
                      'eval': {'tokens': 100000, 'bytes': 400000, 'documents': 10}})
            if prefix:
                add(f'validation-tokens-dclm-10gb-{vocab}', 'tokenize-validation', validation=validation(dataset, vocab))
    state['nodes']['old-held'] = dict(stage='train-model', status='canceled', arguments=[],
        dependencies=[], user_hold=True, hold_reason='User hold', receipt={'workload_id': 'preserve'})
    return state


def test_exact_six_new_runs_with_chains_runtime_and_preserved_history():
    state = fixture()
    original = copy.deepcopy(state)
    cohort = authorize_six_runs(state)
    assert len(cohort['runs']) == len(cohort['required']) == 6
    assert len(state['nodes']) == len(original['nodes']) + 18
    assert all(state['nodes'][k] == v for k, v in original['nodes'].items())
    assert state['study']['training_submission_hold'] == original['study']['training_submission_hold']
    assert set(campaign.ready_nodes(state)) == set(cohort['runs'])
    for key in cohort['runs']:
        node = state['nodes'][key]
        args = submit.parser().parse_args([*campaign_arguments(node), *campaign.placement_arguments(node)])
        submit.validate_args(args)
        task = submit.make_spec(args)['tasks'][0]
        assert args.training_recipe == EMO_RECIPE and args.dataset in ('dolma', 'dclm')
        assert args.learning_rate == .0005 and args.weight_decay == args.embedding_weight_decay == .1
        assert args.batch_size == 128 and args.epochs == 1 and args.seed == 42
        assert not args.health_check and not args.resume_latest
        assert task['context'] == {'priority': 'urgent', 'minRuntime': '1h'}
        assert task['resources']['gpuCount'] == 8
        assert task['constraints']['cluster'] == list(submit.GPU_CLUSTERS)
        assert {'name': 'TOKENIZER_CHAIN_EVALUATION', 'value': '1'} in task['envVars']
        assert not invalid_result({**node, 'gradient_curve_review': {'status': 'unhealthy'}})
        for prefix in ('convert', 'eval'):
            child = state['nodes'][key.replace('train-', prefix + '-', 1)]
            assert child['chain_parent'] == key and child['chain_allocation_gpus'] == 8
            assert child['arguments'][1:] == node['arguments'][1:]
    # No authorization for a seventh run or a changed recipe.
    key = cohort['runs'][0]
    state['nodes']['unapproved'] = copy.deepcopy(state['nodes'][key])
    assert 'unapproved' not in campaign.ready_nodes(state)
    state['nodes'][key]['arguments'] += ['--resume-latest']
    assert key not in campaign.ready_nodes(state)
    with pytest.raises(ValueError, match='already authorized'):
        authorize_six_runs(state)


@pytest.mark.parametrize('failure', ['source', 'validation', 'split'])
def test_prerequisite_mismatch_fails_without_mutating_state(failure):
    state = fixture()
    if failure == 'source':
        state['nodes']['dclm-tokens-10gb-128000']['summary']['revision'] = 'wrong'
    elif failure == 'validation':
        state['nodes']['validation-tokens-dclm-10gb-128000']['summary']['validation']['recipe']['documents_sha256'] = 'wrong'
    else:
        state['nodes']['tokens-10gb-64000']['summary']['meta']['train']['documents'] += 1
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match='mismatch|unverified'):
        authorize_six_runs(state)
    assert state == original


def test_legacy_planner_does_not_reuse_new_emo_recipe():
    state = fixture()
    cohort = authorize_six_runs(state)
    emo = next(k for k in cohort['runs'] if k.startswith('train-dolma-') and '-32000-' in k)
    state['nodes'][emo]['receipt'] = {'workload_id': 'new-emo'}
    result = adaptive_study.model_run(state, '10GB', 'small', 32000, .0005, role='legacy')
    assert result != emo
    assert '--training-recipe' not in state['nodes'][result]['arguments']
