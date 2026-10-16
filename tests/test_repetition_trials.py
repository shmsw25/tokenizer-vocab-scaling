import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import campaign
import report
import submit
from emo_trials import authorize_six_runs
from repetition_filter import FILTER_SETTINGS
from repetition_trials import authorize_three_runs
from test_emo_trials import fixture as source_fixture
from training_recipe import configure_recipe


def fixture():
    state = source_fixture()
    state.update(training_pools=['10GB'], pool_sizes=['10GB'], complete=False)
    for node in state['nodes'].values():
        if node['stage'] in ('tokenize-data', 'tokenize-dclm-data'):
            node.update(data_size='10GB', vocab_size=node['summary']['meta']['vocab_size'])
    original = authorize_six_runs(state)
    for key in original['runs']:
        node = state['nodes'][key]
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/small.json').read_text())
        config['training']['learning_rate'] = .0005
        configure_recipe(config)
        node.update(status='succeeded', receipt={'workload_id': key, 'revision': 'model-source'},
                    summary={'stage': 'train-model', 'revision': 'model-source', 'config': config})
    campaign.prepare_chains(state)
    return state


def test_three_fresh_trials_change_only_filter_and_preserve_all_existing_nodes():
    state = fixture()
    before = copy.deepcopy(state)
    cohort = authorize_three_runs(state)
    assert len(state['nodes']) == len(before['nodes']) + 9
    assert len(cohort['runs']) == len(cohort['required']) == 3
    assert all(state['nodes'][k] == n for k, n in before['nodes'].items())
    assert state['study']['training_submission_hold'] == before['study']['training_submission_hold']
    assert set(cohort['runs']) <= set(campaign.ready_nodes(state))
    for key in cohort['runs']:
        node = state['nodes'][key]
        original = before['nodes'][node['comparison_run']]
        assert node['arguments'] == original['arguments'] + ['--repetition-filter', 'emo-v1']
        assert not node.get('receipt') and not node.get('previous_attempts')
        args = submit.parser().parse_args([*node['arguments'], *campaign.placement_arguments(node)])
        submit.validate_args(args)
        spec = submit.make_spec(args)['tasks'][0]
        assert args.repetition_filter == 'emo-v1' and args.dataset == 'dclm'
        assert args.learning_rate == .0005 and args.epochs == 1 and args.batch_size == 128
        assert spec['context'] == {'priority': 'urgent', 'minRuntime': '1h'}
        assert spec['resources']['gpuCount'] == 8 and spec['constraints']['cluster'] == list(submit.GPU_CLUSTERS)
        assert {'name':'TOKENIZER_CHAIN_EVALUATION','value':'1'} in spec['envVars']
        for prefix in ('convert', 'eval'):
            child = state['nodes'][key.replace('train-', prefix+'-', 1)]
            assert child['chain_parent'] == key and child['arguments'][1:] == node['arguments'][1:]
        assert report.run_config(node)['training']['repetition_filter'] == FILTER_SETTINGS
    data = report.build_report_data(state)
    assert all(r['repetition_filter'] == FILTER_SETTINGS for r in data['runs'] if r['id'] in cohort['runs'])
    key = cohort['runs'][0]
    state['nodes'][key]['arguments'] += ['--max-steps','100']
    assert not campaign.training_submission_allowed(state, key)
    with pytest.raises(ValueError, match='already authorized'):
        authorize_three_runs(state)


@pytest.mark.parametrize('failure', ['receipt','summary','recipe','incomplete'])
def test_changed_comparison_or_source_fails_without_mutation(failure):
    state = fixture()
    source = state['nodes']['dclm-tokens-10gb-128000']
    original = state['nodes']['train-dclm-small-10gb-128000-lr0p0005-emo-dense-v1']
    if failure == 'receipt': source['receipt']['workload_id'] = 'different'
    elif failure == 'summary': source['summary']['meta']['train']['tokens'] += 1
    elif failure == 'recipe': original['arguments'] += ['--repetition-filter','emo-v1']
    else: original['status'] = 'running'
    before = copy.deepcopy(state)
    with pytest.raises(ValueError): authorize_three_runs(state)
    assert state == before
