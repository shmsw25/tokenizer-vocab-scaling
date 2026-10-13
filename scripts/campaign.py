"""Advance a dependency-gated Beaker campaign and render its local HTML report.

Run once per monitoring interval. The state file records an intended job name
before submission, so a lost response can be recovered without duplicate jobs.
Failed jobs require explicit --retry-node after their failure is understood.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import html
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


def log_progress(lines, stage):
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
    return {'text': 'Waiting for measurable progress', 'eta': 'Not yet available'}


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
                    node['assignment'] = assignment
                    # The serialized protobuf omits the empty list for zero GPUs.
                    assigned_gpus = len(assignment.get('gpus', []))
                    node['assigned_gpus'] = assigned_gpus
                    if node['stage'] in RAW_STAGES and assigned_gpus:
                        # The user accepts scheduler-imposed GPU reservations for
                        # this CPU workload so it can use any Weka cluster.
                        node['resource_warning'] = (f'Requested 0 GPUs; Beaker reserved {assigned_gpus}. '
                                                    'Raw-data job remains CPU-only.')
                    elif assigned_gpus != node['expected_gpus']:
                        raise RuntimeError(f"Expected {node['expected_gpus']} GPUs, assigned {assigned_gpus}")
                    else:
                        node.pop('resource_warning', None)
                lines = [{'timestamp': line.timestamp.ToJsonString(),
                          'message': line.message.decode(errors='replace')}
                         for line in b.job.logs(job, tail_lines=160, follow=False)]
                write_json(folder / 'logs' / f'{key}.json', lines)
                node['progress'] = log_progress(lines, node['stage'])
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
    escape = lambda value: html.escape(str(value))
    nodes = state['nodes']
    evaluations = [n for n in nodes.values() if n['stage'] == 'evaluate-model']
    complete = (bool(evaluations) and all(n['status'] == 'succeeded' for n in nodes.values())
                and all(set(EVALS) <= set(n.get('summary', {}).get('metrics', {})) for n in evaluations))
    state['complete'] = complete
    status_counts = Counter(n['status'] for n in nodes.values())
    rows = []
    for key, node in nodes.items():
        receipt = node.get('receipt', {})
        label = escape(key)
        if receipt.get('url'):
            label = f'<a href="{escape(receipt["url"])}">{label}</a>'
        progress = node.get('progress', {})
        error = node.get('monitor_error') or node.get('submission_error') or ''
        if node['status'] == 'pending':
            detail = 'After ' + ', '.join(node['dependencies'])
        else:
            detail = error or progress.get('text', '')
        if node.get('resource_warning'):
            detail += ' ' + node['resource_warning']
        eta = 'Complete' if node['status'] == 'succeeded' else progress.get('eta', 'Awaiting prerequisites')
        rows.append(f'<tr><td>{label}</td><td>{escape(node["status"])}</td>'
                    f'<td>{node["expected_gpus"]} / {escape(node.get("assigned_gpus", "—"))}</td>'
                    f'<td>{escape(detail)}</td><td>{escape(eta)}</td></tr>')
    results = []
    for node in evaluations:
        conversion = nodes[node['dependencies'][0]]
        training = nodes[conversion['dependencies'][0]].get('summary', {})
        cfg = training.get('config', {})
        metrics = node.get('summary', {}).get('metrics', {})
        scores = []
        for task in EVALS:
            value = metrics.get(task, {}).get('bpb' if task == 'dolma_bpb' else 'accuracy')
            scores.append('—' if value is None else (f'{value:.4f}' if task == 'dolma_bpb' else f'{100 * value:.2f}%'))
        results.append('<tr>' + ''.join(f'<td>{escape(value)}</td>' for value in
                       [node.get('data_size', '100GB'), node['model_size'], node['vocab_size'], cfg.get('seq_length', '—'),
                        cfg.get('training', {}).get('learning_rate', '—'), *scores, node['status']]) + '</tr>')
    payload = json.dumps(state, sort_keys=True).replace('<', '\\u003c')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tokenizer vocabulary study</title><style>
body{font:15px/1.55 system-ui,sans-serif;margin:40px auto;padding:0 24px;max-width:1500px;color:#142c3d;background:#f7f9fb}
h1{font-size:34px;line-height:1.2}h2{margin-top:36px}a{color:#08649e}table{border-collapse:collapse;width:100%;background:white}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #dce5eb;vertical-align:top}th{background:#eaf1f6}
.scroll{overflow-x:auto}p{max-width:1100px}.badge{padding:6px 12px;border-radius:20px;background:#dbeef7;display:inline-block}
button{padding:8px 14px;border:1px solid #08649e;border-radius:6px;background:white;color:#08649e;cursor:pointer}
details{margin-top:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}td:nth-child(4){max-width:550px}
</style><body><h1>Tokenizer vocabulary study</h1>'''
    document += f'<p class="badge">{"Complete" if complete else "In progress"} · {escape(state.get("checked_at", now()))}</p>'
    pools = state.get('pool_sizes', ['100GB'])
    training_pools = state.get('training_pools', ['100GB'])
    document += (f'<p>Independent uniform Dolma pools: {escape(", ".join(pools))}. '
                 f'One-epoch model training on {escape(", ".join(training_pools))}; '
                 'BPE vocabularies and model sizes are compared at matched bytes per context within each pool. '
                 'Each training job uses 8 H100s, global batch 256, WSD with 96 warmup steps and a final 10% linear decay. '
                 'Contexts use round(4096 × target tokens/byte ÷ 32k tokens/byte). Per-GPU microbatches: small 32, medium 16, medium_large 8.</p>')
    document += '<h2>Evaluation results</h2><p>Downstream scores use the repository’s byte-normalized scoring (including BLiMP); these differ from standard benchmark scoring. '
    document += 'Higher accuracy is better. Lower held-out Dolma bits/byte is better. Dolma uses the same 1,000 held-out documents for all runs. Dashes are pending results.</p>'
    document += '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in
                ['Pool', 'Model', 'Vocabulary', 'Context', 'LR', 'BLiMP ↑', 'ARC-Easy ↑', 'HellaSwag ↑', 'PIQA ↑', 'Dolma BPB ↓', 'Eval status']) + '</tr></thead><tbody>'
    document += ''.join(results) + '</tbody></table></div>'
    document += '<h2>Raw pools and tokenizers</h2><p>GB labels denote approximate UTF-8 text size from uniform record sampling; token counts are measured, not assumed. Samples are independent across sizes and contain no repetition. Shared evaluation records are excluded from every smaller pool.</p>'
    document += '<div class="scroll"><table><thead><tr><th>Pool</th><th>Actual text GB</th><th>Records</th><th>Vocabulary</th><th>Tokenizer</th><th>Train tokens</th></tr></thead><tbody>'
    for node in nodes.values():
        if node['stage'] != 'train-tokenizer':
            continue
        pool, vocab = node.get('data_size', '100GB'), node['vocab_size']
        info = nodes.get(f'sample-{size_key(pool)}', {}).get('summary', {}).get('pool', {})
        meta = nodes.get(f'tokens-{size_key(pool)}-{vocab}', {}).get('summary', {}).get('meta', {})
        cells = [pool, info.get('actual_text_gb', '—'), info.get('records', '—'), vocab,
                 node['status'], meta.get('train', {}).get('tokens', '—')]
        document += '<tr>' + ''.join(f'<td>{escape(x)}</td>' for x in cells) + '</tr>'
    document += '</tbody></table></div>'
    if state.get('submission_hold'):
        document += '<p>Submissions held: ' + escape(state['submission_hold']) + '</p>'
    document += '<h2>Jobs and ETA</h2><p>' + escape(', '.join(f'{count} {status}' for status, count in sorted(status_counts.items()))) + '</p>'
    document += '<p>ETAs come from current log progress. Queue time, later stages, initialization, and evaluation are not included in training ETAs. '
    document += 'No campaign completion time is claimed before enough throughput is measured. GPU columns show requested / assigned.</p>'
    document += '<div class="scroll"><table><thead><tr><th>Job</th><th>Status</th><th>GPUs</th><th>Progress</th><th>ETA</th></tr></thead><tbody>'
    document += ''.join(rows) + '</tbody></table></div>'
    document += '<h2>Reproducibility</h2><p>Job links retain immutable submitted source snapshots. The exported JSON contains run configurations, checkpoint paths, raw-data and tokenizer provenance, measured context ratios, evaluation counts and margins, and observed resource assignments.</p>'
    document += '<button onclick="download()">Download results JSON</button><details><summary>Run metadata</summary><pre id="metadata"></pre></details>'
    document += f'<script id="data" type="application/json">{payload}</script><script>const state=JSON.parse(document.getElementById("data").textContent);document.getElementById("metadata").textContent=JSON.stringify(state,null,2);function download(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)],{{type:"application/json"}}));a.download="tokenizer-study-results.json";a.click();URL.revokeObjectURL(a.href);}}</script></body></html>'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)


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
