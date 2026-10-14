"""Exercise the actual shell's ordering, GPU visibility, and failure isolation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import campaign
import submit


@pytest.mark.parametrize('failure,expected', [
    ('', ['train-model', 'convert-model', 'evaluate-model']),
    ('train-model', ['train-model']),
    ('convert-model', ['train-model', 'convert-model']),
])
def test_shell_chain(tmp_path, failure, expected):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy(Path(__file__).resolve().parents[1] / 'scripts/run_model.sh', scripts)
    (scripts / 'chain_status.py').write_text('')
    (scripts / 'pipeline.py').write_text('''
import json,os,sys
with open(os.environ['CALLS'], 'a') as f:
 f.write(json.dumps({'stage':sys.argv[1], 'args':sys.argv[2:],
                     'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\\n')
if sys.argv[1]==os.environ['FAILURE']:sys.exit(7)
''')
    calls = tmp_path / 'calls.jsonl'
    env = dict(os.environ, CALLS=str(calls), FAILURE=failure, CUDA_VISIBLE_DEVICES='3,4,5,6,7,8,9,10',
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    arguments = ['--vocab-size', '64000', '--data-size', '5GB', '--batch-size', '128', '--learning-rate', '0.002']
    result = subprocess.run(['bash', str(scripts / 'run_model.sh'), 'train-model', *arguments],
                            env=env, capture_output=True, text=True, timeout=15)
    rows = [json.loads(line) for line in calls.read_text().splitlines()]
    assert result.returncode == (7 if failure else 0), result.stderr
    assert [r['stage'] for r in rows] == expected
    assert all(r['args'] == arguments for r in rows)
    assert rows[0]['cuda'] == env['CUDA_VISIBLE_DEVICES']
    if len(rows) > 1:
        assert rows[1]['cuda'] == ''
    if len(rows) > 2:
        assert rows[2]['cuda'] == '3'


def test_chain_claims_downstream_before_submission_and_reports_partial_failure():
    state = {'campaign': 'test', 'nodes': campaign.make_nodes([32000], ['small'])}
    key = 'train-small-5gb-32000'
    root = state['nodes'][key]
    campaign.prepare_chains(state)
    conversion = state['nodes']['convert-small-5gb-32000']
    evaluation = state['nodes']['eval-small-5gb-32000']
    assert conversion['chain_parent'] == evaluation['chain_parent'] == key
    root.update(status='succeeded', receipt={'workload_id':'shared'})
    assert 'convert-small-5gb-32000' not in campaign.ready_nodes(state)
    snapshot = {'current_stage':'convert-model', 'summaries':{'train-model':{'stage':'train-model'}}}
    campaign.apply_chain_snapshot(state,key,snapshot,'failed')
    assert root['status']=='succeeded' and conversion['status']=='failed' and evaluation['status']=='pending'
    campaign.retry_node(state,'convert-small-5gb-32000')
    campaign.prepare_chains(state)
    assert conversion.get('chain_parent') is None and conversion['expected_gpus']==1
    assert evaluation['chain_parent']=='convert-small-5gb-32000'
    assert root['receipt']['workload_id']=='shared'
    assert 'convert-small-5gb-32000' in campaign.ready_nodes(state)


def test_chained_spec_keeps_training_recipe_and_uses_one_gpu_for_legacy_postprocessing():
    for stage,gpus in [('train-model',8),('convert-model',1)]:
        args=submit.parser().parse_args([stage,'--vocab-size','64000','--chain-evaluation','--cluster','ai2/jupiter'])
        submit.validate_args(args)
        task=submit.make_spec(args)['tasks'][0]
        assert task['resources']['gpuCount']==gpus
        assert {'name':'TOKENIZER_CHAIN_EVALUATION','value':'1'} in task['envVars']
        assert '--chain-evaluation' not in task['arguments']
        assert task['arguments'][0]==stage
