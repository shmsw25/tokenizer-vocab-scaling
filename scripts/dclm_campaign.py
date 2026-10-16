"""Advance the separate DCLM data/tokenizer campaign; never dispatch models."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path

from campaign import now, refresh, retry_node, submit_ready
from pipeline import DEFAULT_ROOT, DCLM_STAGES, REPO, read_json, write_json


def make_state(include_tokenization=False):
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
            if include_tokenization:
                add(f'dclm-tokens-{pool.lower()}-{vocab}', 'tokenize-dclm-data',
                    ['dclm-pools', f'dclm-bpe-{pool.lower()}-{vocab}'], pool, vocab)
    return {'campaign': 'dclmprep-20261014', 'workspace': 'ai2/flex2', 'owner': 'sewonm',
            'created_at': now(), 'scope': 'DCLM raw data and nine tokenizers only; no model training',
            'source': read_json(REPO / 'data/dclm_source.json'), 'nodes': nodes,
            'study': {'gpu_scope': 'all_user', 'gpu_limit': 100}, 'complete': False}


def validate_scope(state):
    expected = make_state(state.get('study', {}).get('tokenization', {}).get('enabled', False))
    if set(state['nodes']) != set(expected['nodes']) or state['source'] != expected['source']:
        raise ValueError('Unexpected DCLM campaign nodes/source; inspect before proceeding')
    for key, node in state['nodes'].items():
        if (node['stage'] not in DCLM_STAGES or node['expected_gpus'] != 0
                or node['arguments'] != expected['nodes'][key]['arguments']
                or node['dependencies'] != expected['nodes'][key]['dependencies']
                or node.get('chain_evaluation') or node.get('model_size') is not None):
            raise ValueError(f'Unauthorized model or changed DCLM stage: {key}')


def enable_tokenization(state):
    """Explicit, idempotent extension; retain every existing node and receipt."""
    validate_scope(state)
    if state.get('study', {}).get('tokenization', {}).get('enabled'):
        return False
    for key, node in make_state(True)['nodes'].items():
        if key not in state['nodes']:
            state['nodes'][key] = node
    state['study']['tokenization'] = {'enabled': True, 'authorized_at': now(),
        'pools': ['5GB', '10GB', '20GB'], 'vocabularies': [32000, 64000, 128000],
        'seed': 42, 'eval_fraction': .01, 'eos_per_document': 1,
        'scope': 'Encode all matching pool documents; no model training or evaluation'}
    state['scope'] = 'DCLM data, nine tokenizers and nine full-pool token streams; no models'
    state['complete'] = False
    state['phase'] = 'tokenization'
    validate_scope(state)
    return True


def collect_progress(state, folder):
    for key, node in state['nodes'].items():
        node.pop('scan_eta_seconds', None)
        node.pop('eta_note', None)
        node.pop('tokenization_eta_seconds', None)
        if node.get('monitor_error') or node['status'] not in ('running', 'succeeded', 'failed', 'canceled'):
            continue
        path = folder / 'logs' / f'{key}.json'
        if not path.exists():
            continue
        observations = []
        tokenization = []
        for line in read_json(path):
            for fragment in line['message'].splitlines():
                if fragment.startswith('DCLM_PROGRESS '):
                    try:
                        observations.append(json.loads(fragment[len('DCLM_PROGRESS '):]))
                    except ValueError:
                        continue
                elif fragment.startswith('TOKENIZATION_PROGRESS '):
                    try:
                        value = json.loads(fragment[len('TOKENIZATION_PROGRESS '):])
                        if tokenization and value['documents'] <= tokenization[-1]['documents']:
                            tokenization = []
                        tokenization.append(value)
                    except (ValueError, KeyError):
                        continue
        if tokenization and node['stage'] == 'tokenize-dclm-data':
            latest = tokenization[-1]
            node['tokenization_progress'] = latest
            pool = state['nodes']['dclm-pools']['summary']['pools'][node['data_size']]
            latest['text_bytes_total'] = pool['text_bytes']
            latest['documents_total'] = pool['records']
            previous = [x for x in tokenization if latest['time'] - x['time'] >= 60]
            if node['status'] == 'running' and previous:
                first = previous[max(0, len(previous) - 5)]
                rate = (latest['text_bytes'] - first['text_bytes']) / (latest['time'] - first['time'])
                if rate > 0:
                    node['tokenization_eta_seconds'] = max(0, pool['text_bytes'] - latest['text_bytes']) / rate
                    node['eta_note'] = 'Measured recent UTF-8 byte throughput; hash verification/finalization extra; queue unknown'
        if observations:
            latest = observations[-1]
            node['dclm_progress'] = latest
            completed = [x for x in observations if x['stage'] == 'download' and 'compressed_bytes_done' in x]
            if completed:
                node['dclm_last_completed_shard_progress'] = completed[-1]
            previous = [x for x in observations if x['stage'] == latest['stage'] and x['time'] < latest['time'] - 60]
            if node['status'] == 'running' and previous and latest['stage'] == 'download' and 'compressed_bytes_done' in latest:
                rate_field = 'new_compressed_bytes_done' if 'new_compressed_bytes_done' in latest else 'compressed_bytes_done'
                first = next((x for x in previous if x.get(rate_field, 0) > 0), None)
                if first and latest[rate_field] > first[rate_field]:
                    rate = (latest[rate_field] - first[rate_field]) / (latest['time'] - first['time'])
                    node['scan_eta_seconds'] = (latest['compressed_bytes_total'] - latest['compressed_bytes_done']) / rate
                    node['eta_note'] = 'Recent new compressed-byte throughput (cache reuse excluded); excludes assembly/pools/tokenizers; variable shard/network cost'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=REPO / '.build/dclm-preparation/campaign.json')
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--enable-tokenization', action='store_true',
                        help='Explicitly extend the existing campaign with nine authorized CPU token streams')
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
        if args.enable_tokenization and not state.get('study', {}).get('tokenization', {}).get('enabled'):
            backup = args.state.parent / 'tokenization-extension' / 'state-before.json'
            if backup.exists():
                raise ValueError('Tokenization extension backup exists; inspect interrupted migration before retry')
            write_json(backup, state)
            enable_tokenization(state)
            write_json(backup.with_name('migration.json'), {'at': now(),
                'backup_sha256': hashlib.sha256(backup.read_bytes()).hexdigest(),
                'added_nodes': [k for k,n in state['nodes'].items() if n['stage'] == 'tokenize-dclm-data']})
            write_json(args.state, state)
        if not args.offline:
            refresh(state, args.state.parent)
            collect_progress(state, args.state.parent)
        for key in args.retry_node:
            progress = {field: state['nodes'][key][field] for field in
                        ('dclm_progress', 'dclm_last_completed_shard_progress', 'scan_eta_seconds',
                         'tokenization_progress', 'tokenization_eta_seconds', 'eta_note')
                        if field in state['nodes'][key]}
            retry_node(state, key)
            if state['nodes'][key].get('previous_attempts'):
                state['nodes'][key]['previous_attempts'][-1].update(progress)
            for field in progress:
                state['nodes'][key].pop(field, None)
        write_json(args.state, state)
        if args.advance:
            submit_ready(state, args.state, max_submissions=args.max_submissions)
        state.pop('tokenization_verification_error', None)
        if state.get('study', {}).get('tokenization', {}).get('enabled'):
            try:
                for node in state['nodes'].values():
                    if node['stage'] == 'tokenize-dclm-data' and node['status'] == 'succeeded':
                        if (node['summary']['stage'] != node['stage']
                                or node['summary']['revision'] != node['receipt']['revision']):
                            raise ValueError('DCLM tokenization worker/source identity mismatch')
                from report import dclm_training_tokens
                dclm_training_tokens(state)
            except (ValueError, KeyError, TypeError) as error:
                state['tokenization_verification_error'] = f'{type(error).__name__}: {error}'
        state['complete'] = all(n['status'] == 'succeeded' and n.get('summary') and not n.get('monitor_error')
                                for n in state['nodes'].values()) and not state.get('tokenization_verification_error')
        state['phase'] = ('complete' if state['complete'] else 'download' if state['nodes']['dclm-download']['status'] != 'succeeded'
                          else 'pools' if state['nodes']['dclm-pools']['status'] != 'succeeded'
                          else 'tokenizers' if any(n['stage'] == 'train-dclm-tokenizer' and n['status'] != 'succeeded' for n in state['nodes'].values())
                          else 'tokenization')
        write_json(args.state, state)
        model_state = args.state.parent.parent / 'campaign.json'
        if model_state.exists() and not state.get('tokenization_verification_error'):
            from report import render_report
            render_report(read_json(model_state), model_state.with_name('report.html'))
        print(json.dumps({'checked_at': state.get('checked_at'), 'phase': state['phase'], 'complete': state['complete'],
                          'tokenization_verification_error': state.get('tokenization_verification_error'),
                          'counts': dict(Counter(n['status'] for n in state['nodes'].values())),
                          'nodes': {k: {f: n[f] for f in ('status', 'receipt', 'hostname', 'assigned_gpus', 'dclm_progress',
                                    'scan_eta_seconds', 'tokenization_progress', 'tokenization_eta_seconds', 'eta_note', 'monitor_error', 'submission_error', 'admission_wait') if f in n}
                                    for k,n in state['nodes'].items()}}, indent=2))


if __name__ == '__main__':
    main()
