import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import campaign
import report
import submit
from dolma_emo_trials import authorize_runs, COHORT
from dolma_emo_report import comparison_data
from pipeline import EVALS
from training_recipe import campaign_arguments
from test_emo_trials import fixture as original_fixture


def fixture():
    state = original_fixture()
    state.update(complete=False, training_pools=['5GB', '10GB', '20GB'], pool_sizes=['5GB', '10GB', '20GB'])
    state['study'].update(dolma_validation_id='dolma100m-v1', fixed_learning_rates={'medium_large': .001})
    for vocab in (32000, 64000, 128000):
        old = state['nodes'][f'tokens-10gb-{vocab}']
        old.update(data_size='10GB', vocab_size=vocab)
        new = copy.deepcopy(old)
        new['data_size'] = '5GB'
        new['receipt']['workload_id'] = f'tokens-5gb-{vocab}'
        state['nodes'][f'tokens-5gb-{vocab}'] = new
        streams = state['nodes']['prepare-validation']['summary']['streams']
        streams[f'default_5GB_{vocab}'] = copy.deepcopy(streams[f'default_10GB_{vocab}'])
        streams[f'default_5GB_{vocab}']['recipe']['pool'] = '5GB'
    # This fixture's DCLM streams are unrelated to the new comparison.
    for vocab in (32000, 64000, 128000):
        state['nodes'][f'dclm-tokens-10gb-{vocab}'].update(data_size='10GB', vocab_size=vocab, dataset='dclm')
    return state


def test_36_exact_recipes_preserve_old_nodes_and_holds():
    state = fixture()
    original = copy.deepcopy(state)
    cohort = authorize_runs(state)
    assert len(cohort['runs']) == len(cohort['required']) == 36
    assert len(state['nodes']) == len(original['nodes']) + 108
    assert all(state['nodes'][k] == v for k, v in original['nodes'].items())
    assert state['study']['training_submission_hold'] == original['study']['training_submission_hold']
    assert set(campaign.ready_nodes(state)) == set(cohort['runs'])
    persisted = json.loads(json.dumps(state, sort_keys=True))
    assert campaign.ready_nodes(persisted) == cohort['runs']
    coordinates = set()
    for key in cohort['runs']:
        node = state['nodes'][key]
        args = submit.parser().parse_args([*campaign_arguments(node), *campaign.placement_arguments(node)])
        submit.validate_args(args)
        task = submit.make_spec(args)['tasks'][0]
        coordinates.add((args.data_size, args.model_size, args.vocab_size, args.embedding_weight_decay))
        assert args.dataset == 'dolma' and args.training_recipe == 'emo-dense-v1'
        assert args.repetition_filter == 'emo-v1' and args.learning_rate == .0005
        assert args.weight_decay == .1 and args.embedding_weight_decay in (0., .1)
        assert args.batch_size == 128 and args.epochs == 1 and args.seed == 42
        assert not args.resume_latest
        assert task['context'] == {'priority': 'urgent', 'minRuntime': '1h' if args.data_size == '5GB' else '3h'}
        assert task['resources']['gpuCount'] == 8
        assert task['constraints']['cluster'] == list(submit.GPU_CLUSTERS)
        assert {'name': 'TOKENIZER_CHAIN_EVALUATION', 'value': '1'} in task['envVars']
        expected_micro = 4 if args.model_size == 'medium_large' and args.vocab_size == 128000 else 8 if args.model_size == 'medium_large' else 16
        assert args.micro_batch_size == expected_micro
        for prefix in ('convert', 'eval'):
            child = state['nodes'][key.replace('train-', prefix + '-', 1)]
            assert child['chain_parent'] == key and child['chain_allocation_gpus'] == 8
            assert child['arguments'][1:] == node['arguments'][1:]
    assert len(coordinates) == 36
    key = cohort['runs'][0]
    state['nodes']['unapproved'] = copy.deepcopy(state['nodes'][key])
    assert 'unapproved' not in campaign.ready_nodes(state)
    args = state['nodes'][key]['arguments']
    args[args.index('--learning-rate') + 1] = '.001'
    # Even a future exception cannot bypass the user-set LR ceiling.
    state['study']['training_submission_exceptions'][key]['arguments_sha256'] = hashlib.sha256(json.dumps(args, separators=(',', ':')).encode()).hexdigest()
    assert not campaign.training_submission_allowed(state, key)
    with pytest.raises(ValueError, match='already authorized'):
        authorize_runs(state)


@pytest.mark.parametrize('failure', ['source', 'validation', 'split'])
def test_bad_sources_fail_before_mutation(failure):
    state = fixture()
    if failure == 'source':
        state['nodes']['tokens-5gb-128000']['summary']['revision'] = 'wrong'
    elif failure == 'validation':
        state['nodes']['prepare-validation']['summary']['streams']['default_5GB_128000']['recipe']['documents_sha256'] = 'wrong'
    else:
        state['nodes']['tokens-10gb-64000']['summary']['meta']['train']['documents'] += 1
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match='mismatch|unverified'):
        authorize_runs(state)
    assert state == original


def test_source_receipt_change_blocks_admission_without_releasing_hold():
    state = fixture()
    cohort = authorize_runs(state)
    state['nodes']['tokens-5gb-32000']['receipt']['workload_id'] = 'different-job'
    assert not any(campaign.training_submission_allowed(state, key) for key in cohort['runs'])
    assert state['study']['training_submission_hold']


def test_report_keeps_both_embed_wd_results_and_requires_complete_evaluations():
    state = fixture()
    cohort = authorize_runs(state)
    assert report.build_report_data(state)['runs'] == []
    data = comparison_data(state)
    assert len(data['runs']) == 36 and data['selected'] == []
    assert len(data['panels']) == 4
    assert data['token_vocab_labels'] == ['32K', '64K', '128K']
    assert {r['dataset'] for r in data['training_tokens']} == {'Dolma'}
    keys = [k for k in cohort['runs'] if 'medium_large-10gb-128000' in k]
    for i, key in enumerate(keys):
        state['nodes'][key]['status'] = 'succeeded'
        state['nodes'][key.replace('train-', 'convert-', 1)]['status'] = 'succeeded'
        evaluation = state['nodes'][key.replace('train-', 'eval-', 1)]
        evaluation['summary'] = dict(metrics={m: {'bpb' if m == 'dolma_bpb' else 'accuracy': .5} for m in EVALS},
            shared_validation={'metrics': {'validation_bpb': .9 + i * .01}})
    # A partial same-shell evaluation must not leak benchmark metrics into graphs.
    assert comparison_data(state)['selected'] == []
    for key in keys:
        state['nodes'][key.replace('train-', 'eval-', 1)]['status'] = 'succeeded'
    data = comparison_data(state)
    assert set(data['selected']) == set(keys)  # Legacy medium_large LR1e-3 rule does not apply.
    points = [y for fig in data['figures']['dolma_bpb'] for trace in fig['data'] for y in trace['y']]
    assert sorted(points) == [.9, .91]
    for key in keys:
        run = next(r for r in data['runs'] if r['id'] == key)
        assert run['compute']['epoch_flops'] > 0
    state['nodes'][keys[0]]['loss_health_decision'] = {'source': 'user', 'status': 'unhealthy'}
    assert keys[0] not in comparison_data(state)['selected']
    assert report.build_report_data(state)['runs'] == []
