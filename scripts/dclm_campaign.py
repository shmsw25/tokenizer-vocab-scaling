"""Advance the separate DCLM data/tokenizer campaign; never dispatch models."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path

from campaign import now, refresh, retry_node, submit_ready
from pipeline import DEFAULT_ROOT, DCLM_STAGES, REPO, read_json, write_json


def make_state():
    nodes = {}
    def add(key, stage, dependencies, pool=None, vocab=None):
        arguments = [stage, '--root', DEFAULT_ROOT]
        if vocab:
            arguments += ['--data-size', pool, '--vocab-size', str(vocab)]
        nodes[key] = {'stage': stage, 'arguments': arguments, 'dependencies': dependencies,
                      'data_size': pool, 'vocab_size': vocab, 'model_size': None, 'status': 'pending',
                      'expected_gpus': 0, 'placement_clusters': ['ai2/saturn', 'ai2/neptune'],
                      'allow_gpu_reservation': True, 'priority': 'normal',
                      'min_runtime': '1h' if pool == '5GB' else '3h'}
    add('dclm-download', 'download-dclm', [])
    add('dclm-pools', 'sample-dclm', ['dclm-download'])
    for pool in ('5GB', '10GB', '20GB'):
        for vocab in (32000, 64000, 128000):
            add(f'dclm-bpe-{pool.lower()}-{vocab}', 'train-dclm-tokenizer', ['dclm-pools'], pool, vocab)
    return {'campaign': 'dclmprep-20261014', 'workspace': 'ai2/flex2', 'owner': 'sewonm',
            'created_at': now(), 'scope': 'DCLM raw data and nine tokenizers only; no model training',
            'source': read_json(REPO / 'data/dclm_source.json'), 'nodes': nodes,
            'study': {'gpu_scope': 'all_user', 'gpu_limit': 100}, 'complete': False}


def validate_scope(state):
    expected = make_state()
    if set(state['nodes']) != set(expected['nodes']) or state['source'] != expected['source']:
        raise ValueError('Unexpected DCLM campaign nodes/source; inspect before proceeding')
    for key, node in state['nodes'].items():
        if (node['stage'] not in DCLM_STAGES or node['expected_gpus'] != 0
                or node['arguments'] != expected['nodes'][key]['arguments']
                or node['dependencies'] != expected['nodes'][key]['dependencies']
                or node.get('chain_evaluation') or node.get('model_size') is not None):
            raise ValueError(f'Unauthorized model or changed DCLM stage: {key}')


def collect_progress(state, folder):
    for key, node in state['nodes'].items():
        path = folder / 'logs' / f'{key}.json'
        if not path.exists():
            continue
        observations = []
        for line in read_json(path):
            for fragment in line['message'].splitlines():
                if fragment.startswith('DCLM_PROGRESS '):
                    try:
                        observations.append(json.loads(fragment[len('DCLM_PROGRESS '):]))
                    except ValueError:
                        continue
        if observations:
            latest = observations[-1]
            node['dclm_progress'] = latest
            previous = [x for x in observations if x['stage'] == latest['stage'] and x['time'] < latest['time'] - 60]
            if previous and latest['stage'] == 'download' and 'compressed_bytes_done' in latest:
                first = next((x for x in previous if 'compressed_bytes_done' in x), None)
                if first and latest['compressed_bytes_done'] > first['compressed_bytes_done']:
                    rate = (latest['compressed_bytes_done'] - first['compressed_bytes_done']) / (latest['time'] - first['time'])
                    node['scan_eta_seconds'] = (latest['compressed_bytes_total'] - latest['compressed_bytes_done']) / rate
                    node['eta_note'] = 'Recent compressed-byte throughput; excludes assembly/pools/tokenizers; variable shard/network cost'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=REPO / '.build/dclm-preparation/campaign.json')
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--advance', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--max-submissions', type=int)
    parser.add_argument('--retry-node', action='append', default=[],
                        help='Only after diagnosing a terminal failure; preserves the previous receipt')
    args = parser.parse_args()
    if args.offline and args.advance:
        parser.error('Offline operation cannot submit')
    if args.retry_node and (args.offline or not args.advance):
        parser.error('--retry-node requires live --advance')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    # Serialize admission with the original study as well as this state file.
    with (REPO / '.build/campaign.lock').open('a') as global_lock, args.state.with_suffix('.lock').open('a') as handle:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.init:
            if args.state.exists():
                parser.error('DCLM campaign already exists; omit --init')
            write_json(args.state, make_state())
        state = read_json(args.state)
        validate_scope(state)
        if not args.offline:
            refresh(state, args.state.parent)
            collect_progress(state, args.state.parent)
        for key in args.retry_node:
            retry_node(state, key)
        write_json(args.state, state)
        if args.advance:
            submit_ready(state, args.state, max_submissions=args.max_submissions)
        state['complete'] = all(n['status'] == 'succeeded' and n.get('summary') and not n.get('monitor_error')
                                for n in state['nodes'].values())
        state['phase'] = ('complete' if state['complete'] else 'download' if state['nodes']['dclm-download']['status'] != 'succeeded'
                          else 'pools' if state['nodes']['dclm-pools']['status'] != 'succeeded' else 'tokenizers')
        write_json(args.state, state)
        print(json.dumps({'checked_at': state.get('checked_at'), 'phase': state['phase'], 'complete': state['complete'],
                          'counts': dict(Counter(n['status'] for n in state['nodes'].values())),
                          'nodes': {k: {f: n[f] for f in ('status', 'receipt', 'hostname', 'assigned_gpus', 'dclm_progress',
                                    'scan_eta_seconds', 'eta_note', 'monitor_error', 'submission_error', 'admission_wait') if f in n}
                                    for k,n in state['nodes'].items()}}, indent=2))


if __name__ == '__main__':
    main()
