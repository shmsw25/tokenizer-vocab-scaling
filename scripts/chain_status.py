"""Small log snapshots keep completed stages visible while a chained job runs."""
import argparse
import json
from pathlib import Path
import time

STAGES = ('train-model', 'convert-model', 'evaluate-model')


def snapshot(stage, results=Path('/results')):
    summaries = {}
    for name in STAGES:
        path = results / f'{name}.json'
        if path.is_file():
            summaries[name] = json.loads(path.read_text())
    result = {'current_stage': stage, 'summaries': summaries}
    branches = results / 'decay_branches.json'
    if branches.is_file():
        result['branch_checkpoints'] = json.loads(branches.read_text())
    for field, filename in [('runtime_batch', 'runtime_batch_plan.json'),
                            ('runtime_batch_verified', 'runtime_batch_verified.json')]:
        path = results / filename
        if path.is_file():
            result[field] = json.loads(path.read_text())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        if args.watch:
            time.sleep(30)
        print('PIPELINE_CHAIN ' + json.dumps(snapshot(args.stage), separators=(',', ':')), flush=True)
        if not args.watch:
            break
