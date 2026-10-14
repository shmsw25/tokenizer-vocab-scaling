"""Audit W&B loss curves; invalidate clear explosions and flag gray areas."""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


POLICY = {
    'version': 2, 'metric': 'train/CE loss', 'warmup_grace_steps': 100,
    'baseline_observations': 100, 'loss_multiple': 1.5, 'minimum_increase': 1.0,
    'sustained_consecutive_observations': 5,
    'episode_gap_steps': 100, 'severe_loss_multiple': 3.0,
    'severe_minimum_increase': 3.0, 'severe_consecutive_observations': 3,
    'action': 'invalidate_clear_explosions_review_borderline',
}


def invalid_result(node):
    """A human decision takes precedence over the automatic curve assessment."""
    decision = node.get('loss_health_decision', {}).get('status')
    if decision is not None:
        return decision == 'unhealthy'
    if node.get('health_stop', {}).get('classification') == 'unhealthy':
        return True
    return node.get('loss_curve_review', {}).get('status') == 'unhealthy'


def spike_flags(value, reference):
    """Shared thresholds for the full-history auditor and in-job loss guard."""
    spike = value >= POLICY['loss_multiple'] * reference and value - reference >= POLICY['minimum_increase']
    severe = (spike and value >= POLICY['severe_loss_multiple'] * reference
              and value - reference >= POLICY['severe_minimum_increase'])
    return spike, severe


def analyze_history(rows, warmup_steps=192):
    """Compare each post-warmup loss to the preceding 100 finite observations.

    This is an operational validity policy, not a statistical test.
    Missing/null W&B values are not evidence of NaN training loss.
    """
    points = {}
    missing = 0
    for row in rows:
        step, value = row.get('_step'), row.get('train/CE loss')
        if not isinstance(step, (int, float)) or not math.isfinite(step):
            continue
        if not isinstance(value, (int, float)):
            missing += 1
            continue
        points[int(step)] = float(value)
    baseline = deque(maxlen=POLICY['baseline_observations'])
    spikes, nonfinite = [], []
    severe_streak, longest_severe_streak = 0, 0
    spike_streak, longest_spike_streak = 0, 0
    previous_step = None
    for step, value in sorted(points.items()):
        if not math.isfinite(value):
            nonfinite.append(step)
            severe_streak = 0
            spike_streak = 0
            continue
        severe = spike = False
        if step >= warmup_steps + POLICY['warmup_grace_steps'] and len(baseline) == baseline.maxlen:
            reference = statistics.median(baseline)
            spike, severe = spike_flags(value, reference)
            if spike:
                spikes.append({'step': step, 'loss': value, 'preceding_median': reference,
                               'multiple': value / reference if reference > 0 else None})
        # Require adjacent optimizer steps, so sparse observations cannot manufacture persistence.
        severe_streak = (severe_streak + 1 if previous_step == step - 1 else 1) if severe else 0
        longest_severe_streak = max(longest_severe_streak, severe_streak)
        spike_streak = (spike_streak + 1 if previous_step == step - 1 else 1) if spike else 0
        longest_spike_streak = max(longest_spike_streak, spike_streak)
        previous_step = step
        baseline.append(value)
    episodes = []
    for point in spikes:
        if not episodes or point['step'] - episodes[-1]['last_step'] > POLICY['episode_gap_steps']:
            episodes.append({'first_step': point['step'], 'last_step': point['step'],
                             'observations': 0, 'peak': point})
        episode = episodes[-1]
        episode['last_step'] = point['step']
        episode['observations'] += 1
        if point['loss'] > episode['peak']['loss']:
            episode['peak'] = point
    finite = [(step, v) for step, v in sorted(points.items()) if math.isfinite(v)]
    return {'policy': dict(POLICY), 'warmup_steps': warmup_steps,
            'status': ('unhealthy' if (longest_severe_streak >= POLICY['severe_consecutive_observations']
                                      or longest_spike_streak >= POLICY['sustained_consecutive_observations'])
                       else 'review_needed' if spikes or nonfinite else 'no_spike_detected' if finite else 'unavailable'),
            'observations': len(finite), 'missing_values': missing,
            'first_step': finite[0][0] if finite else None,
            'last_step': finite[-1][0] if finite else None,
            'nonfinite_steps': nonfinite, 'spike_observations': len(spikes),
            'longest_severe_streak': longest_severe_streak,
            'longest_spike_streak': longest_spike_streak, 'episodes': episodes}


def history_paths(node):
    """Include pre-resume training; recovery does not erase earlier instability."""
    return list(dict.fromkeys(p for source in [*node.get('previous_attempts', []), node]
        if (p := source.get('reported_metrics', {}).get('wandb_path'))))


def logged_wandb_links(lines):
    """Only return links actually published by this Beaker job."""
    found = {}
    pattern = r'https://(?:wandb\.ai/|forge\.coreweave\.com/wandb/)([\w.-]+/[\w.-]+)/runs/([\w-]+)'
    for line in lines:
        for match in re.finditer(pattern, line['message']):
            found[f'{match[1]}/{match[2]}'] = match[0]
    return found


def beaker_history_rows(lines):
    """Use explicit train_step values; validation echoes cannot create steps."""
    points = {}
    for line in lines:
        for fragment in line['message'].splitlines():
            if 'REPORT_METRICS ' not in fragment:
                continue
            try:
                metrics = json.loads(fragment.split('REPORT_METRICS ', 1)[1])
            except (ValueError, TypeError):
                continue
            if not isinstance(metrics, dict):
                continue
            step, loss = metrics.get('train_step'), metrics.get('train_ce')
            if (not isinstance(step, int) or isinstance(step, bool)
                    or not isinstance(loss, (int, float)) or isinstance(loss, bool)):
                continue
            if step in points and points[step] != loss:
                raise ValueError('Conflicting training losses for one optimizer step')
            points[step] = loss
    return [{'_step': step, 'train/CE loss': loss} for step, loss in sorted(points.items())]


def cache_beaker_history(lines, cache_dir, workload_id, job_id, checked_at):
    rows = beaker_history_rows(lines)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (hashlib.sha256(job_id.encode()).hexdigest() + '.json')
    payload = {'workload_id': workload_id, 'job_id': job_id,
               'fetched_at': checked_at, 'rows': rows}
    temporary = cache.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, sort_keys=True) + '\n')
    temporary.replace(cache)
    return {'kind': 'beaker', 'workload_id': workload_id, 'job_id': job_id,
            'url': f'https://beaker.org/ex/{workload_id}', 'history_file': str(cache),
            'history_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}


def discover_beaker_history(node, beaker, cache_dir, checked_at):
    """Recover published W&B links or per-step CE without creating W&B runs."""
    receipt_id = node.get('receipt', {}).get('workload_id')
    terminal = node['status'] in ('succeeded', 'failed', 'canceled')
    prior = node.get('wandb_discovery', {})
    if (terminal and prior.get('receipt_id') == receipt_id and prior.get('complete')
            and all(Path(x['history_file']).exists() for x in node.get('beaker_loss_histories', []))):
        return
    histories, attempts = [], []
    seen = set()
    for source in [*node.get('previous_attempts', []), node]:
        wid = source.get('receipt', {}).get('workload_id')
        if not wid or wid in seen:
            continue
        seen.add(wid)
        status = source.get('job_status', {})
        if status.get('status') == 'STATUS_CANCELED' and not status.get('started'):
            attempts.append({'workload_id': wid, 'status': 'never_started'})
            continue
        try:
            workload = beaker.workload.get(wid)
            for task in workload.experiment.tasks:
                for job in beaker.job.list(task=task):
                    if not job.status.HasField('started'):
                        continue
                    lines = [{'message': line.message.decode(errors='replace')}
                             for line in beaker.job.logs(job, follow=False)]
                    links = logged_wandb_links(lines)
                    # Multiple segments remain explicit, rather than assigning an arbitrary one.
                    if len(links) == 1 and not source.get('reported_metrics', {}).get('wandb_path'):
                        path, url = next(iter(links.items()))
                        source.setdefault('reported_metrics', {}).update(wandb_path=path, wandb_url=url)
                    if wid == receipt_id and source is not node:
                        raise ValueError('Current receipt also belongs to a previous attempt')
                    histories.append(cache_beaker_history(lines, cache_dir, wid, job.id, checked_at))
                    attempts.append({'workload_id': wid, 'job_id': job.id, 'wandb_links': links})
        except Exception as error:
            attempts.append({'workload_id': wid, 'error': type(error).__name__})
    # Preserve any earlier source if a temporary API failure prevents fetching it.
    by_job = {x['job_id']: x for x in node.get('beaker_loss_histories', [])}
    by_job.update({x['job_id']: x for x in histories})
    node['beaker_loss_histories'] = list(by_job.values())
    report_to = node.get('summary', {}).get('config', {}).get('training', {}).get('report_to')
    node['wandb_discovery'] = {'receipt_id': receipt_id, 'checked_at': checked_at,
        'complete': terminal and any(x['workload_id'] == receipt_id for x in histories),
        'status': 'found' if history_paths(node) else 'disabled' if report_to == 'none' else 'not_found',
        'report_to': report_to, 'attempts': attempts}


def beaker_history_hashes(node):
    return [x['history_sha256'] for x in node.get('beaker_loss_histories', [])]


def needs_review(node):
    review = node.get('loss_curve_review', {})
    return (review.get('policy_version') != POLICY['version']
            or review.get('paths') != history_paths(node)
            or review.get('beaker_history_hashes', []) != beaker_history_hashes(node)
            or not review.get('terminal') or review.get('error'))


def review_node(node, api, cache_dir, checked_at, current_run=None):
    paths = history_paths(node)
    terminal = node['status'] in ('succeeded', 'failed', 'canceled')
    current_path = node.get('reported_metrics', {}).get('wandb_path')
    warmup = node.get('summary', {}).get('config', {}).get('training', {}).get('warmup_steps', 192)
    sources = []
    for path in paths:
        # Never use a W&B path as a filesystem path, and never store auth material.
        cache = Path(cache_dir) / (hashlib.sha256(path.encode()).hexdigest() + '.json')
        immutable = terminal or path != current_path
        cached = json.loads(cache.read_text()) if cache.exists() else {}
        if immutable and cached.get('terminal') and cached.get('path') == path:
            rows = cached['rows']
        else:
            run = current_run if path == current_path and current_run is not None else api.run(path)
            rows = [{k: row.get(k) for k in ('_step', '_timestamp', 'train/CE loss')}
                    for row in run.scan_history(keys=['_step', '_timestamp', 'train/CE loss'], page_size=1000)]
            cache.parent.mkdir(parents=True, exist_ok=True)
            payload = {'path': path, 'terminal': immutable, 'fetched_at': checked_at, 'rows': rows}
            temporary = cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, sort_keys=True) + '\n')
            temporary.replace(cache)
        entity, project, run_id = path.split('/')
        sources.append({'kind': 'wandb', 'path': path, 'url': f'https://wandb.ai/{entity}/{project}/runs/{run_id}',
                        'history_file': str(cache), 'history_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
                        **analyze_history(rows, warmup)})
    for evidence in node.get('beaker_loss_histories', []):
        cache = Path(evidence['history_file'])
        if hashlib.sha256(cache.read_bytes()).hexdigest() != evidence['history_sha256']:
            raise ValueError('Beaker loss-history hash mismatch')
        payload = json.loads(cache.read_text())
        if (payload['job_id'], payload['workload_id']) != (evidence['job_id'], evidence['workload_id']):
            raise ValueError('Beaker loss-history identity mismatch')
        sources.append({**evidence, **analyze_history(payload['rows'], warmup)})
    available = any(x['observations'] for x in sources)
    status = ('unhealthy' if any(x['status'] == 'unhealthy' for x in sources)
              else 'review_needed' if any(x['status'] == 'review_needed' for x in sources)
              else 'no_spike_detected' if available else 'unavailable')
    node['loss_curve_review'] = {'policy_version': POLICY['version'], 'status': status,
        'checked_at': checked_at, 'paths': paths, 'terminal': terminal, 'sources': sources,
        'beaker_history_hashes': beaker_history_hashes(node)}
    # No mutation of receipts, training status, metrics, LR, or human decisions.
