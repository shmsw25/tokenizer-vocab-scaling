"""Advance a dependency-gated Beaker campaign and render its local HTML report.

Run once per monitoring interval. The state file records an intended job name
before submission, so a lost response can be recovered without duplicate jobs.
Failed jobs require explicit --retry-node after their failure is understood.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

from pipeline import (DEFAULT_ROOT, EVALS, MODEL_MICROBATCHES, POOL_SIZES, TRAINING_POOLS,
                      RAW_STAGES, REPO, data_size, read_json, write_json)

TERMINAL = {'succeeded', 'failed', 'canceled'}
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def now():
    return datetime.now(timezone.utc).isoformat()


def size_key(size):
    return size.lower().replace('.', 'p')


def make_nodes(vocabs, models, pools=POOL_SIZES, training_pools=TRAINING_POOLS):
    if 32000 not in vocabs:
        raise ValueError('Include 32000 as the shared context reference')
    if not pools or not set(training_pools) <= set(pools):
        raise ValueError('Every model-training pool must also be a tokenizer pool')
    if len(set(pools)) != len(pools) or len(set(training_pools)) != len(training_pools):
        raise ValueError('Pool lists must not contain duplicates')
    nodes = {}

    def add(key, stage, dependencies, vocab=None, model=None, pool=None):
        arguments = [stage, '--root', DEFAULT_ROOT]
        if stage == 'sample-pool':
            arguments += ['--data-size', pool, '--sampling-seed', '42']
        if vocab is not None:
            arguments += ['--vocab-size', str(vocab), '--data-size', pool]
        if model is not None:
            arguments += ['--model-size', model, '--gpus', '8', '--batch-size', '256',
                          '--micro-batch-size', str(MODEL_MICROBATCHES[model]), '--epochs', '1']
        nodes[key] = {'stage': stage, 'dependencies': dependencies, 'arguments': arguments,
                      'vocab_size': vocab, 'model_size': model, 'data_size': pool, 'status': 'pending',
                      'expected_gpus': 8 if stage == 'train-model' else int(stage == 'evaluate-model')}

    add('download', 'download-data', [])
    sample_keys = [f'sample-{size_key(pool)}' for pool in pools]
    for pool, sample_key in zip(pools, sample_keys):
        add(sample_key, 'sample-pool', ['download'], pool=pool)
        for vocab in vocabs:
            suffix = f'{size_key(pool)}-{vocab}'
            # Finish all requested raw pools before beginning tokenizer training.
            add(f'bpe-{suffix}', 'train-tokenizer', sample_keys, vocab, pool=pool)
            if pool not in training_pools:
                continue
            add(f'tokens-{suffix}', 'tokenize-data', [f'bpe-{suffix}'], vocab, pool=pool)
            for model in models:
                run = f'{model}-{suffix}'
                add(f'train-{run}', 'train-model', sorted({f'tokens-{suffix}',
                    f'tokens-{size_key(pool)}-32000'}), vocab, model, pool)
                add(f'convert-{run}', 'convert-model', [f'train-{run}'], vocab, model, pool)
                add(f'eval-{run}', 'evaluate-model', [f'convert-{run}'], vocab, model, pool)
    return nodes


def migrate_pools(state, vocabs, models, pools, training_pools):
    """Replace only the unsubmitted legacy plan, retaining the live parent job."""
    if state.get('plan_version') == 2:
        raise ValueError('Pool plan already migrated; do not reinitialize or duplicate jobs')
    old = state['nodes']
    for key, node in old.items():
        if key != 'download' and (node['status'] != 'pending' or node.get('receipt') or node.get('name')):
            raise ValueError(f'Cannot replace submitted legacy stage {key}; inspect its actual job first')
    nodes = make_nodes(vocabs, models, pools, training_pools)
    nodes['download'] = old['download']
    state.setdefault('plan_history', []).append({'changed_at': now(), 'nodes': old,
        'reason': 'Six independent raw pools, 18 tokenizers, and 27 one-epoch model runs'})
    state.update(nodes=nodes, plan_version=2, pool_sizes=list(pools), training_pools=list(training_pools))
    state.pop('submission_hold', None)


def ready_nodes(state):
    if state.get('submission_hold'):
        return []
    return [key for key, node in state['nodes'].items()
            if node['status'] == 'pending' and not node.get('submission_error')
            and all(state['nodes'][dep]['status'] == 'succeeded' for dep in node['dependencies'])]


def log_progress(lines, stage, total_records=None):
    # Beaker can pack thousands of carriage-return tqdm updates into one log
    # message. Select the latest complete update, including after a phase change.
    lines = [{'timestamp': line['timestamp'], 'message': ANSI.sub('', fragment).strip()}
             for line in lines for fragment in re.split(r'[\r\n]+', line['message']) if fragment.strip()]
    for line in reversed(lines):
        text = line['message']
        match = re.search(r'\[step=(\d+)/(\d+),epoch=[^,\]]+(?:,eta=([^\]]+))?\]', text)
        if match:
            return {'step': int(match[1]), 'total_steps': int(match[2]),
                    'text': f'Step {match[1]}/{match[2]}',
                    'eta': match[3] or 'Not yet estimated', 'observed_at': line['timestamp']}
        if match := re.search(r'\[([^\[\]]*<[^\[\]]*)\]', text):
            eta = match[1].split('<', 1)[1].split(',')[0]
            if text.startswith('Counting records:'):
                eta += ' for record counting; sample writing follows'
            elif text.startswith('Sampling records:'):
                eta += ' for sample writing'
            return {'text': text[-250:], 'eta': eta, 'observed_at': line['timestamp']}
        if stage == 'sample-pool':
            if text.startswith(('Pool ready:', 'Pool already complete:')):
                return {'text': text, 'eta': 'Sample complete; finalizing job',
                        'observed_at': line['timestamp']}
            if re.match(r'Sampling [\d.]+GB: [\d,]+/[\d,]+ eligible records;', text):
                # Beaker may buffer the following tqdm line until its newline.
                # Do not fall back to the previous phase's 100%/zero ETA.
                return {'text': text, 'eta': 'Unknown; waiting for sample-writing progress',
                        'observed_at': line['timestamp']}
        if stage == 'download-data':
            if text.startswith(('Raw data ready:', 'Sample contains ')):
                return {'text': text, 'eta': 'Sample complete; finalizing job',
                        'observed_at': line['timestamp']}
            if 'repeat_dolma.py' in text or (text.startswith('Sampling ') and
                                           not text.startswith('Sampling records:')):
                return {'text': '250GB download complete; preparing the uniform 100GB sample',
                        'eta': 'Sampling ETA not yet available', 'observed_at': line['timestamp']}
            if re.search(r'Downloaded [\d,]+ records, ([\d.]+) / 250 GB', text):
                break  # Current phase is still downloading; estimate its rate below.
    if stage == 'download-data':
        points = []
        for line in lines:
            match = re.search(r'Downloaded [\d,]+ records, ([\d.]+) / 250 GB', line['message'])
            if match:
                points.append((datetime.fromisoformat(line['timestamp'].replace('Z', '+00:00')).timestamp(), float(match[1])))
        if points:
            first, last = points[0], points[-1]
            eta = 'Need more progress samples; README estimates about a day for download/sample'
            if last[0] - first[0] > 10 and last[1] > first[1]:
                hours = max(0, 250 - last[1]) / (last[1] - first[1]) * (last[0] - first[0]) / 3600
                eta = f'~{hours:.1f} h for download at recent rate; sampling follows'
            return {'text': f'{last[1]:.3f}/250 GB text downloaded', 'eta': eta,
                    'observed_at': datetime.fromtimestamp(last[0], timezone.utc).isoformat()}
    if stage == 'tokenize-data':
        points = []
        for line in lines:
            if match := re.fullmatch(r'([\d,]+) documents', line['message']):
                point = (datetime.fromisoformat(line['timestamp'].replace('Z', '+00:00')).timestamp(),
                         int(match[1].replace(',', '')))
                if points and point[1] <= points[-1][1]:
                    points = []  # A restarted attempt must establish its own rate.
                points.append(point)
        if points:
            first, last = points[max(0, len(points) - 4)], points[-1]
            text = f'{last[1]:,} documents tokenized'
            eta = 'Need pool size and more progress samples'
            if total_records:
                text = f'{last[1]:,}/{total_records:,} documents ({100 * last[1] / total_records:.1f}%)'
                if last[0] - first[0] > 10 and last[1] > first[1]:
                    minutes = max(0, total_records - last[1]) * (last[0] - first[0]) / (last[1] - first[1]) / 60
                    eta = f'~{minutes:.0f} min at recent rate as of last update; finalization extra'
            return {'text': text, 'eta': eta, 'documents': last[1], 'total_documents': total_records,
                    'observed_at': datetime.fromtimestamp(last[0], timezone.utc).isoformat()}
    return {'text': 'Waiting for measurable progress', 'eta': 'Not yet available'}


def record_assignment(node, assignment):
    node['assignment'] = assignment
    # The protobuf omits the empty list for zero GPUs.
    assigned_gpus = len(assignment.get('gpus', []))
    node['assigned_gpus'] = assigned_gpus
    allow_reservation = node['stage'] in RAW_STAGES or node.get('allow_gpu_reservation', False)
    if node['expected_gpus'] == 0 and allow_reservation and assigned_gpus:
        node['resource_warning'] = (f'Requested 0 GPUs; Beaker reserved {assigned_gpus}. '
                                    'Job remains CPU-only with CUDA disabled.')
    elif assigned_gpus != node['expected_gpus']:
        raise RuntimeError(f"Expected {node['expected_gpus']} GPUs, assigned {assigned_gpus}")
    else:
        node.pop('resource_warning', None)


def placement_arguments(node):
    result = []
    if node.get('placement_clusters'):
        result += ['--cluster', *node['placement_clusters']]
    if node.get('allow_gpu_reservation'):
        result += ['--allow-gpu-reservation']
    return result


def refresh(state, folder):
    from beaker import Beaker, BeakerWorkloadStatus
    from beaker.exceptions import BeakerWorkloadNotFound
    from google.protobuf.json_format import MessageToDict
    with Beaker.from_env(default_workspace=state['workspace'], check_for_upgrades=False) as b:
        for key, node in state['nodes'].items():
            if not node.get('receipt') and not node.get('name'):
                continue
            if node['status'] in TERMINAL and (node.get('summary') or key == 'download'):
                continue
            try:
                ref = node.get('receipt', {}).get('workload_id') or f"{state['owner']}/{node['name']}"
                try:
                    workload = b.workload.get(ref)
                except BeakerWorkloadNotFound:
                    if node.get('receipt'):
                        raise
                    node['submission_not_found_at'] = now()
                    continue  # no experiment was created before the interrupted submission
                node.pop('submission_not_found_at', None)
                if not node.get('receipt'):
                    node['receipt'] = {'workload_id': workload.experiment.id,
                                       'url': b.experiment.url(workload), 'name': node['name']}
                node.pop('submission_error', None)
                node['status'] = BeakerWorkloadStatus(workload.status).name
                node['checked_at'] = now()
                job = b.workload.get_latest_job(workload)
                detail = MessageToDict(job)
                node['job_id'] = job.id
                node['job_status'] = detail.get('status', {})
                assigned_details = detail.get('assignmentDetails', {})
                assignment = assigned_details.get('resourceAssignment')
                node['hostname'] = next((entry.get('literal') for entry in
                                         assigned_details.get('assignedEnvironmentVariables', [])
                                         if entry['name'] == 'BEAKER_NODE_HOSTNAME'), None)
                if assignment is not None:
                    record_assignment(node, assignment)
                lines = [{'timestamp': line.timestamp.ToJsonString(),
                          'message': line.message.decode(errors='replace')}
                         for line in b.job.logs(job, tail_lines=160, follow=False)]
                write_json(folder / 'logs' / f'{key}.json', lines)
                if node['stage'] == 'train-model':
                    for line in lines:
                        for fragment in line['message'].splitlines():
                            if 'REPORT_METRICS ' in fragment:
                                try:
                                    metrics = json.loads(fragment.split('REPORT_METRICS ', 1)[1])
                                    if isinstance(metrics, dict) and metrics.get('step', 0) >= node.get('reported_metrics', {}).get('step', 0):
                                        node['reported_metrics'] = metrics
                                except (ValueError, TypeError):
                                    pass  # A truncated log line is not a metric observation.
                pool = state['nodes'].get(f"sample-{size_key(node.get('data_size') or '100GB')}", {})
                total_records = pool.get('summary', {}).get('pool', {}).get('records')
                node['progress'] = log_progress(lines, node['stage'], total_records)
                if node['status'] == 'queued':
                    node['progress'] = {'text': node['job_status'].get('failedSchedulingMessage', 'Queued'),
                                        'eta': 'Unknown queue wait'}
                if node['status'] == 'succeeded' and key != 'download':
                    dataset = b.workload.get_results(workload)
                    url = b.dataset.get_file_link(dataset, 'summary.json')
                    with urllib.request.urlopen(url, timeout=60) as response:
                        node['summary'] = json.loads(response.read(10_000_000))
                    write_json(folder / 'summaries' / f'{key}.json', node['summary'])
                node.pop('monitor_error', None)
            except Exception as error:
                node['monitor_error'] = f'{type(error).__name__}: {error}'
        state['checked_at'] = now()


def submit_ready(state, path):
    for key in ready_nodes(state):
        node = state['nodes'][key]
        if any(state['nodes'][dep].get('monitor_error') for dep in node['dependencies']):
            continue
        node.setdefault('name', f"{state['campaign']}-{key.replace('_', '-')}")
        # Persist intent before starting any network call. If interrupted,
        # refresh recovers the experiment by name before attempting submission.
        node['submission_error'] = 'Submission in progress or outcome not yet recovered'
        write_json(path, state)
        receipt_path = path.parent / 'receipts' / f'{key}.json'
        command = [sys.executable, str(REPO / 'scripts/submit.py'), *node['arguments'],
                   *placement_arguments(node),
                   '--workspace', state['workspace'], '--name', node['name'],
                   '--receipt', str(receipt_path), '--spec-file', str(path.parent / 'specs' / f'{key}.json')]
        try:
            subprocess.run(command, cwd=REPO, check=True)
            node['receipt'] = read_json(receipt_path)
            node['status'] = 'submitted'
            node.pop('submission_error', None)
        except Exception as error:
            node['submission_error'] = f'{type(error).__name__}: {error}'
        write_json(path, state)


def retry_node(state, key):
    node = state['nodes'][key]
    if (node['status'] == 'pending' and node.get('submission_error')
            and node.get('submission_not_found_at') and not node.get('receipt')):
        # Keep the same deterministic name after a live lookup proved that no
        # experiment exists. The ordinary submit path will persist intent again.
        node.pop('submission_error')
        node.pop('submission_not_found_at')
        return
    if node['status'] not in {'failed', 'canceled'}:
        raise ValueError('Retry is allowed only for a confirmed failed or canceled experiment')
    if any(n['status'] != 'pending' and key in n['dependencies'] for n in state['nodes'].values()):
        raise ValueError('Cannot retry a job after its dependents have been submitted')
    history = node.setdefault('previous_attempts', [])
    if len(history) >= 2:
        raise ValueError('Already attempted three times; investigate before creating a new campaign')
    history.append({k: node[k] for k in ('receipt', 'job_status', 'progress') if k in node})
    for field in ('receipt', 'job_id', 'job_status', 'progress', 'assignment', 'assigned_gpus',
                  'monitor_error', 'submission_error', 'summary', 'resource_warning', 'hostname'):
        node.pop(field, None)
    node['status'] = 'pending'
    node['name'] = f"{state['campaign']}-{key.replace('_', '-')}-retry{len(history)}"


def render_report(state, destination):
    from report import render_report as render
    render(state, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=REPO / '.build/campaign.json')
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--download-receipt', type=Path)
    parser.add_argument('--campaign', default='tokstudy-20261012')
    parser.add_argument('--vocabs', nargs='+', type=int, default=[32000, 64000, 128000])
    parser.add_argument('--models', nargs='+', choices=MODEL_MICROBATCHES, default=list(MODEL_MICROBATCHES))
    parser.add_argument('--pool-sizes', nargs='+', type=data_size, default=list(POOL_SIZES))
    parser.add_argument('--training-pools', nargs='+', type=data_size, default=list(TRAINING_POOLS))
    parser.add_argument('--migrate-pools', action='store_true',
                        help='Replace unsubmitted legacy stages while preserving the existing 100GB job')
    parser.add_argument('--workspace', default='ai2/flex2')
    parser.add_argument('--owner', default='sewonm')
    parser.add_argument('--advance', action='store_true', help='Submit stages whose prerequisites have succeeded')
    parser.add_argument('--retry-node', help='Retry a diagnosed failure or a confirmed missing submission (requires --advance)')
    parser.add_argument('--offline', action='store_true', help='Render saved state without Beaker access')
    args = parser.parse_args()
    args.state = args.state.resolve()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.init:
            if args.state.exists():
                parser.error('State already exists; omit --init to resume')
            if not re.fullmatch(r'[a-z0-9-]+', args.campaign):
                parser.error('Campaign name must contain lowercase letters, digits, and hyphens')
            state = {'campaign': args.campaign, 'workspace': args.workspace, 'owner': args.owner,
                     'created_at': now(), 'plan_version': 2, 'pool_sizes': args.pool_sizes,
                     'training_pools': args.training_pools,
                     'nodes': make_nodes(args.vocabs, args.models, args.pool_sizes, args.training_pools)}
            if args.download_receipt:
                receipt = read_json(args.download_receipt)
                state['nodes']['download'].update(receipt=receipt, name=receipt['name'], status='submitted')
            write_json(args.state, state)
        else:
            state = read_json(args.state)
        if not args.offline:
            refresh(state, args.state.parent)
        if args.migrate_pools:
            migrate_pools(state, args.vocabs, args.models, args.pool_sizes, args.training_pools)
        if args.retry_node:
            if not args.advance or args.offline:
                parser.error('--retry-node requires --advance and live status checking')
            retry_node(state, args.retry_node)
        write_json(args.state, state)
        if args.advance:
            if args.offline:
                parser.error('--offline cannot submit jobs')
            # No live gRPC client exists while git archive/submission subprocesses run.
            submit_ready(state, args.state)
        report = args.state.parent / 'report.html'
        render_report(state, report)
        write_json(args.state, state)
        counts = Counter(node['status'] for node in state['nodes'].values())
        print(json.dumps({'checked_at': state.get('checked_at'), 'complete': state['complete'],
                          'counts': counts, 'report': str(report),
                          'active': {key: {field: node[field] for field in
                                     ('status', 'progress', 'hostname', 'assigned_gpus', 'resource_warning',
                                      'monitor_error', 'submission_error', 'receipt') if field in node}
                                     for key, node in state['nodes'].items() if node['status'] not in {'pending', 'succeeded'}}}, indent=2))


if __name__ == '__main__':
    main()
