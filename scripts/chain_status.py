"""Small log snapshots keep completed stages visible while a chained job runs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

STAGES = ('train-model', 'convert-model', 'evaluate-model')


def log_records(value):
    """Keep each record below Beaker's 16KiB log-fragment boundary."""
    payload = json.dumps(value, separators=(',', ':')).encode()
    if len(payload) < 8000:
        return ['PIPELINE_CHAIN ' + payload.decode()]
    identity = hashlib.sha256(payload).hexdigest()
    chunks = [payload[i:i + 5000] for i in range(0, len(payload), 5000)]
    return ['PIPELINE_CHAIN_PART ' + json.dumps({
        'sha256': identity, 'index': i, 'count': len(chunks),
        'payload': base64.b64encode(chunk).decode()}, separators=(',', ':'))
        for i, chunk in enumerate(chunks)]


def read_log_snapshots(lines):
    """Accept complete hash-verified snapshots; retain warnings for damaged logs.

    Old allocations can emit large lines into which Beaker inserts a timestamp.
    Their final result artifacts remain authoritative after successful exit.
    """
    groups, snapshots, warnings = {}, [], []
    for line in lines:
        for fragment in line['message'].splitlines():
            if not fragment.startswith(('PIPELINE_CHAIN ', 'PIPELINE_CHAIN_PART ')):
                continue
            try:
                kind, text = fragment.split(' ', 1)
                item = json.loads(text)
                if kind == 'PIPELINE_CHAIN_PART':
                    identifier, index, count = item['sha256'], item['index'], item['count']
                    if not isinstance(count, int) or not 1 <= count <= 1024 or not 0 <= index < count:
                        raise ValueError('Invalid snapshot chunk bounds')
                    group = groups.setdefault(identifier, {'count': count, 'parts': {}})
                    if group['count'] != count:
                        raise ValueError('Snapshot chunk count changed')
                    group['parts'][index] = base64.b64decode(item['payload'], validate=True)
                    if len(group['parts']) != count:
                        continue
                    payload = b''.join(group['parts'][i] for i in range(count))
                    if hashlib.sha256(payload).hexdigest() != identifier:
                        raise ValueError('Snapshot checksum mismatch')
                    item = json.loads(payload)
                    del groups[identifier]
                if not isinstance(item, dict) or item.get('current_stage') not in STAGES or not isinstance(item.get('summaries'), dict):
                    raise ValueError('Invalid snapshot structure')
                snapshots.append(item)
            except (ValueError, KeyError, TypeError) as error:
                warnings.append({'timestamp': line.get('timestamp'), 'kind': fragment.split(' ', 1)[0],
                                 'error': type(error).__name__, 'characters': len(fragment)})
    return snapshots, warnings[-20:]


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
        for record in log_records(snapshot(args.stage)):
            print(record, flush=True)
        if not args.watch:
            break
