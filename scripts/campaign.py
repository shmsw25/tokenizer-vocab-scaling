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

from pipeline import (DEFAULT_ROOT, DEFAULT_BATCH_SIZE, EVALS, MODEL_STAGES, MODEL_MICROBATCHES, POOL_SIZES, TRAINING_POOLS,
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
            lr = read_json(REPO / 'configs' / f'{model}.json')['training']['learning_rate']
            arguments += ['--model-size', model, '--gpus', '8', '--batch-size', str(DEFAULT_BATCH_SIZE),
                          '--micro-batch-size', str(MODEL_MICROBATCHES[model]), '--epochs', '1',
                          '--learning-rate', str(lr)]
        nodes[key] = {'stage': stage, 'dependencies': dependencies, 'arguments': arguments,
                      'vocab_size': vocab, 'model_size': model, 'data_size': pool, 'status': 'pending',
                      'expected_gpus': 8 if stage == 'train-model' else int(stage == 'evaluate-model')}
        if stage == 'train-model':
            nodes[key].update(priority='urgent', placement_clusters=['ai2/jupiter'],
                              min_runtime='8h' if vocab == 32000 else '0s', chain_evaluation=True)

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


def record_validation_publications(node, lines):
    """Retain atomic publication notices while the CPU job builds other streams."""
    workload = node.get('receipt', {}).get('workload_id')
    if node.get('validation_publications', {}).get('workload_id') != workload:
        node['validation_publications'] = {'workload_id': workload, 'streams': {}}
    published = node['validation_publications']
    for line in lines:
        for fragment in line['message'].splitlines():
            if fragment.startswith('VALIDATION_READY '):
                published['corpus'] = json.loads(fragment.removeprefix('VALIDATION_READY '))
            elif fragment.startswith('VALIDATION_TOKENS_READY '):
                meta = json.loads(fragment.removeprefix('VALIDATION_TOKENS_READY '))
                recipe = meta['recipe']
                published['streams'][f"{recipe['pool']}:{recipe['vocab_size']}"] = meta


def dependency_ready(state, node, dependency):
    prerequisite = state['nodes'][dependency]
    if prerequisite['status'] == 'succeeded':
        return True
    study = state.get('study', {})
    if (not study.get('validation_overlap', {}).get('enabled')
            or dependency != study.get('validation_node')
            or prerequisite['status'] != 'running' or prerequisite.get('monitor_error')
            or node['stage'] not in (*MODEL_STAGES, 'validate-model')):
        return False
    published = prerequisite.get('validation_publications', {})
    if published.get('workload_id') != prerequisite.get('receipt', {}).get('workload_id'):
        return False  # Never reuse readiness from a superseded preparation attempt.
    corpus = published.get('corpus', {})
    stream = published.get('streams', {}).get(f"{node.get('data_size')}:{node.get('vocab_size')}", {})
    return bool(corpus and stream
                and stream['recipe']['validation_id'] == study.get('validation_id')
                and stream['recipe']['documents_sha256'] == corpus['sha256']
                and stream['documents'] == corpus['documents']
                and stream['bytes'] == corpus['text_bytes'])


def ready_nodes(state):
    from adaptive_study import excluded_run
    if state.get('submission_hold'):
        return []
    ready = [key for key, node in state['nodes'].items()
            if node['status'] == 'pending' and not node.get('submission_error') and not node.get('chain_parent')
            and not node.get('hold_reason') and not excluded_run(state, node)
            and all(dependency_ready(state, node, dep) for dep in node['dependencies'])]
    return sorted(ready, key=lambda key: '--resume-latest' not in state['nodes'][key]['arguments'])


def prepare_chains(state):
    """Claim downstream nodes before submission, including ambiguous network outcomes."""
    nodes = state['nodes']
    for key, node in nodes.items():
        if not node.get('chain_evaluation') or node.get('chain_parent'):
            continue
        parent = key
        allocated = node['expected_gpus']
        for stage in MODEL_STAGES[MODEL_STAGES.index(node['stage']) + 1:]:
            child_key, child = next((k, n) for k, n in nodes.items()
                                    if n['stage'] == stage and parent in n['dependencies'])
            if stage in node.get('chain_detached_stages', []):
                parent = child_key
                continue
            if child.get('receipt') and child.get('chain_parent') != key:
                raise ValueError(f'Cannot chain already-submitted independent stage {child_key}')
            child['chain_parent'] = key
            child['chain_allocation_gpus'] = allocated
            if node.get('receipt'):
                child['receipt'] = dict(node['receipt'])
            parent = child_key


def apply_chain_snapshot(state, key, snapshot, workload_status):
    """Track stage completion separately from the shared job's lifetime."""
    root = state['nodes'][key]
    root['chain_snapshot'] = snapshot
    root['chain_job_status'] = workload_status
    if snapshot.get('branch_checkpoints'):
        root['branch_checkpoints'] = snapshot['branch_checkpoints']
    for child_key, node in state['nodes'].items():
        if child_key != key and node.get('chain_parent') != key:
            continue
        node['receipt'] = dict(root['receipt'])
        for field in ('job_id', 'job_status', 'hostname', 'assigned_gpus', 'assignment', 'checked_at'):
            if field in root:
                node[field] = root[field]
        summary = snapshot.get('summaries', {}).get(node['stage'])
        if summary is not None:
            if summary.get('stage') != node['stage']:
                raise ValueError('Chained stage summary mismatch')
            node.update(status='succeeded', summary=summary)
            if node['stage'] == 'train-model':
                metrics = summary.get('training', {}).get('latest_metrics', {})
                if metrics.get('step', 0) >= node.get('reported_metrics', {}).get('step', 0):
                    node['reported_metrics'] = metrics
        elif node['stage'] == snapshot.get('current_stage'):
            node['status'] = workload_status if workload_status != 'succeeded' else 'failed'
            if workload_status == 'succeeded':
                node['monitor_error'] = 'Chained job exited successfully without a required stage summary'
        else:
            node['status'] = 'pending'
        if summary is not None or workload_status != 'succeeded':
            node.pop('monitor_error', None)


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
    expected = node.get('chain_allocation_gpus', node['expected_gpus'])
    if expected == 0 and allow_reservation and assigned_gpus:
        node['resource_warning'] = (f'Requested 0 GPUs; Beaker reserved {assigned_gpus}. '
                                    'Job remains CPU-only with CUDA disabled.')
    elif assigned_gpus != expected:
        raise RuntimeError(f"Expected {expected} GPUs, assigned {assigned_gpus}")
    else:
        node.pop('resource_warning', None)


def placement_arguments(node):
    result = []
    if node.get('chain_evaluation'):
        result += ['--chain-evaluation']
    if node.get('priority'):
        result += ['--priority', node['priority']]
    if node.get('min_runtime') is not None:
        result += ['--min-runtime', node['min_runtime']]
    if node.get('placement_clusters'):
        result += ['--cluster', *node['placement_clusters']]
    if node.get('allow_gpu_reservation'):
        result += ['--allow-gpu-reservation']
    if node.get('wandb_secret'):
        result += ['--wandb-secret', node['wandb_secret']]
    return result


def refresh(state, folder):
    from beaker import Beaker, BeakerWorkloadStatus
    from beaker.exceptions import BeakerWorkloadNotFound
    from google.protobuf.json_format import MessageToDict
    with Beaker.from_env(default_workspace=state['workspace'], check_for_upgrades=False) as b:
        for key, node in state['nodes'].items():
            if node.get('chain_parent'):
                continue  # The owner refreshes every stage from this one job.
            if not node.get('receipt') and not node.get('name'):
                continue
            if (node['status'] in TERMINAL and (node.get('summary') or key == 'download')
                    and (not node.get('chain_evaluation') or node.get('chain_job_status') in TERMINAL)):
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
                workload_status = node['status']
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
                         for line in b.job.logs(job, tail_lines=2000 if node.get('chain_evaluation') else 160, follow=False)]
                write_json(folder / 'logs' / f'{key}.json', lines)
                if node['stage'] == 'prepare-validation':
                    record_validation_publications(node, lines)
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
                            for prefix, field in [('WSD_BRANCH_READY ', 'branch_checkpoints'),
                                                  ('HEALTH_STOP ', 'health_stop'),
                                                  ('RESUME_VERIFIED ', 'resume_verified')]:
                                if fragment.startswith(prefix):
                                    node[field] = json.loads(fragment[len(prefix):])
                pool = state['nodes'].get(f"sample-{size_key(node.get('data_size') or '100GB')}", {})
                total_records = pool.get('summary', {}).get('pool', {}).get('records')
                node['progress'] = log_progress(lines, node['stage'], total_records)
                if node['status'] == 'queued':
                    node['progress'] = {'text': node['job_status'].get('failedSchedulingMessage', 'Queued'),
                                        'eta': 'Unknown queue wait'}
                if node.get('chain_evaluation'):
                    snapshot = node.get('chain_snapshot', {'current_stage': node['stage'], 'summaries': {}})
                    for line in lines:
                        for fragment in line['message'].splitlines():
                            if fragment.startswith('PIPELINE_CHAIN '):
                                snapshot = json.loads(fragment[len('PIPELINE_CHAIN '):])
                    if workload_status == 'succeeded':
                        dataset = b.workload.get_results(workload)
                        for stage in MODEL_STAGES[MODEL_STAGES.index(node['stage']):]:
                            url = b.dataset.get_file_link(dataset, f'{stage}.json')
                            with urllib.request.urlopen(url, timeout=60) as response:
                                snapshot['summaries'][stage] = json.loads(response.read(10_000_000))
                    apply_chain_snapshot(state, key, snapshot, workload_status)
                    for child_key, child in state['nodes'].items():
                        if (child_key == key or child.get('chain_parent') == key) and child.get('summary'):
                            write_json(folder / 'summaries' / f'{child_key}.json', child['summary'])
                elif node['status'] == 'succeeded' and key != 'download':
                    dataset = b.workload.get_results(workload)
                    result_file = 'archive.json' if node['stage'] == 'archive-model-directory' else 'summary.json'
                    url = b.dataset.get_file_link(dataset, result_file)
                    with urllib.request.urlopen(url, timeout=60) as response:
                        node['summary'] = json.loads(response.read(10_000_000))
                    write_json(folder / 'summaries' / f'{key}.json', node['summary'])
                node.pop('monitor_error', None)
            except Exception as error:
                node['monitor_error'] = f'{type(error).__name__}: {error}'
        state['checked_at'] = now()


def submit_ready(state, path, max_submissions=None):
    submitted = 0
    for key in ready_nodes(state):
        if max_submissions is not None and submitted >= max_submissions:
            break
        node = state['nodes'][key]
        if any(state['nodes'][dep].get('monitor_error') for dep in node['dependencies']):
            continue
        if state.get('study'):
            from adaptive_study import live_gpu_budget, reservation
            try:
                budget = live_gpu_budget(state)
                state['gpu_budget'] = budget
                state.pop('gpu_budget_error', None)
            except Exception as error:
                state['gpu_budget_error'] = f'{type(error).__name__}: {error}'
                write_json(path, state)
                return  # Fail closed: unknown allocations cannot justify a launch.
            if budget['reserved'] + reservation(node) > budget['limit']:
                node['admission_wait'] = 'Waiting for capacity under the 100-GPU cap (queued reservations included)'
                write_json(path, state)
                continue
            node.pop('admission_wait', None)
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
            submitted += 1
            node.pop('submission_error', None)
            prepare_chains(state)
        except Exception as error:
            node['submission_error'] = f'{type(error).__name__}: {error}'
        write_json(path, state)


def retry_node(state, key):
    from adaptive_study import excluded_run
    node = state['nodes'][key]
    if excluded_run(state, node):
        raise ValueError('Cannot retry a run with a learning rate excluded by the user')
    if node.get('chain_job_status') and node['chain_job_status'] not in TERMINAL:
        raise ValueError('Cannot retry a stage while its shared job is still active')
    if node.get('health_stop') and '--resume-latest' not in node['arguments']:
        raise ValueError('Health-stopped runs require an explicitly authorized checkpoint resume')
    if node.get('chain_parent'):
        owner = state['nodes'][node['chain_parent']]
        if owner.get('chain_job_status') not in TERMINAL:
            raise ValueError('Cannot retry a stage while its shared job is still active')
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
    if sum(attempt.get('kind') not in {'reconfiguration', 'scheduling_change'} for attempt in history) >= 2:
        raise ValueError('Already attempted three times; investigate before creating a new campaign')
    history.append({k: node[k] for k in ('receipt', 'job_status', 'progress', 'health_stop',
                   'reported_metrics', 'wandb_observation', 'arguments', 'resume_authorization') if k in node})
    if node['stage'] == 'train-model' and node.get('chain_evaluation'):
        # A stopped training chain has not executed its children; retain their old receipts.
        for child in state['nodes'].values():
            if child.get('chain_parent') == key:
                if child.get('receipt'):
                    child.setdefault('previous_attempts', []).append({'receipt': child['receipt'],
                        'kind': 'reconfiguration', 'reason': 'Parent training resumed from checkpoint'})
                for field in ('receipt', 'job_id', 'job_status', 'assignment', 'assigned_gpus', 'hostname'):
                    child.pop(field, None)
    if node.get('chain_parent'):
        owner_key = node.pop('chain_parent')
        owner = state['nodes'][owner_key]
        detached = MODEL_STAGES[MODEL_STAGES.index(node['stage']):]
        owner['chain_detached_stages'] = sorted(set(owner.get('chain_detached_stages', [])) | set(detached))
        node.pop('chain_allocation_gpus', None)
        if node['stage'] == 'convert-model':
            node.update(chain_evaluation=True, expected_gpus=1, placement_clusters=['ai2/jupiter'])
            node.pop('allow_gpu_reservation', None)
        for child in state['nodes'].values():
            if child.get('chain_parent') == owner_key and child['stage'] in detached:
                if child.get('receipt'):
                    child.setdefault('previous_attempts', []).append({'receipt': child['receipt'],
                        'kind': 'reconfiguration', 'reason': 'Retry stopped chain from its failed stage'})
                for field in ('receipt', 'name', 'chain_parent', 'chain_allocation_gpus'):
                    child.pop(field, None)
                child['status'] = 'pending'
    for field in ('receipt', 'job_id', 'job_status', 'progress', 'assignment', 'assigned_gpus',
                  'monitor_error', 'submission_error', 'summary', 'resource_warning', 'hostname',
                  'chain_snapshot', 'chain_job_status', 'health_stop', 'wandb_final_checked',
                  'wandb_observation', 'resume_verified'):
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
    parser.add_argument('--retry-node', action='append', help='Retry a diagnosed terminal failure; repeat for multiple nodes (requires --advance)')
    parser.add_argument('--max-submissions', type=int, help='Limit this advance to N submissions for a verified rollout')
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
        prepare_chains(state)
        if not args.offline:
            refresh(state, args.state.parent)
        if state.get('study'):
            from adaptive_study import advance_plan, monitor_wandb
            if not args.offline:
                monitor_wandb(state)
            advance_plan(state)
        if args.migrate_pools:
            migrate_pools(state, args.vocabs, args.models, args.pool_sizes, args.training_pools)
        if args.retry_node:
            if not args.advance or args.offline:
                parser.error('--retry-node requires --advance and live status checking')
            for key in args.retry_node:
                retry_node(state, key)
        write_json(args.state, state)
        if args.advance:
            if args.offline:
                parser.error('--offline cannot submit jobs')
            # No live gRPC client exists while git archive/submission subprocesses run.
            submit_ready(state, args.state, max_submissions=args.max_submissions)
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
