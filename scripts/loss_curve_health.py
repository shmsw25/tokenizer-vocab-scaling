"""Audit W&B loss curves; invalidate clear explosions and flag gray areas."""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
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
    return node.get('loss_curve_review', {}).get('status') == 'unhealthy'


def analyze_history(rows, warmup_steps=192):
    """Compare each post-warmup loss to the preceding 100 finite observations.

    This is a review heuristic, not a statistical test or an early-stop rule.
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
            if value >= POLICY['loss_multiple'] * reference and value - reference >= POLICY['minimum_increase']:
                spike = True
                spikes.append({'step': step, 'loss': value, 'preceding_median': reference,
                               'multiple': value / reference if reference > 0 else None})
                severe = (value >= POLICY['severe_loss_multiple'] * reference
                          and value - reference >= POLICY['severe_minimum_increase'])
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


def needs_review(node):
    review = node.get('loss_curve_review', {})
    return (review.get('policy_version') != POLICY['version']
            or review.get('paths') != history_paths(node)
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
        sources.append({'path': path, 'url': f'https://wandb.ai/{entity}/{project}/runs/{run_id}',
                        'history_file': str(cache), 'history_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
                        **analyze_history(rows, warmup)})
    available = any(x['observations'] for x in sources)
    status = ('unhealthy' if any(x['status'] == 'unhealthy' for x in sources)
              else 'review_needed' if any(x['status'] == 'review_needed' for x in sources)
              else 'no_spike_detected' if available else 'unavailable')
    node['loss_curve_review'] = {'policy_version': POLICY['version'], 'status': status,
        'checked_at': checked_at, 'paths': paths, 'terminal': terminal, 'sources': sources}
    # No mutation of receipts, training status, metrics, LR, or human decisions.
