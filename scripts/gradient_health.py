"""User-approved gradient-norm validity policy, independent of CE loss."""
import hashlib
import json
import math
from pathlib import Path
import statistics


POLICY = {'version': 1, 'metric': 'optim/total grad norm',
          'warmup_grace_steps': 100, 'baseline_observations': 100,
          'review_multiple': 5.0, 'unhealthy_multiple': 10.0,
          'action': 'invalidate_single_10x_spike_review_5_to_10x'}


def observe(state, step, value, warmup_steps=192):
    """One optimizer observation; shared by offline review and the native guard.

    The metric is the total norm before clipping. Missing values are not NaNs.
    Each observation uses only the preceding finite observations, never itself.
    """
    if (not isinstance(step, (int, float)) or isinstance(step, bool) or not math.isfinite(step)
            or int(step) != step or step <= state.get('last_step', -1)):
        return None
    state['last_step'] = int(step)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    point = {'step': int(step), 'norm': value if math.isfinite(value) else None}
    if not math.isfinite(value) or value < 0:
        return {**point, 'status': 'review_needed', 'nonfinite': not math.isfinite(value)}
    values = state.setdefault('values', [])
    point['status'] = 'no_spike_detected'
    if step >= warmup_steps + POLICY['warmup_grace_steps'] and len(values) == POLICY['baseline_observations']:
        reference = statistics.median(values)
        multiple = value / reference if reference > 0 else None
        point.update(preceding_median=reference, multiple=multiple)
        if multiple is not None and multiple >= POLICY['unhealthy_multiple']:
            point['status'] = 'unhealthy'
        elif (multiple is not None and multiple >= POLICY['review_multiple']) or (reference == 0 and value > 0):
            point['status'] = 'review_needed'
    state['values'] = (values + [float(value)])[-POLICY['baseline_observations']:]
    return point


def analyze_history(rows, warmup_steps=192):
    points, missing = {}, 0
    for row in rows:
        step, value = row.get('_step'), row.get(POLICY['metric'])
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            missing += 1
            continue
        if (isinstance(step, (int, float)) and not isinstance(step, bool)
                and math.isfinite(step) and int(step) == step):
            if step in points and points[step] != value and not (math.isnan(points[step]) and math.isnan(value)):
                raise ValueError('Conflicting gradient norms at one optimizer step')
            points[int(step)] = value
    state, spikes, invalid_steps, finite_steps = {}, [], [], []
    for step, value in sorted(points.items()):
        point = observe(state, step, value, warmup_steps)
        if point.get('nonfinite') or value < 0:
            invalid_steps.append(step)
        else:
            finite_steps.append(step)
            if point['status'] != 'no_spike_detected':
                spikes.append(point)
    return {'policy': dict(POLICY), 'warmup_steps': warmup_steps,
            'status': ('unhealthy' if any(p['status'] == 'unhealthy' for p in spikes)
                       else 'review_needed' if spikes or invalid_steps
                       else 'no_spike_detected' if finite_steps else 'unavailable'),
            'observations': len(finite_steps), 'missing_values': missing,
            'first_step': min(finite_steps) if finite_steps else None,
            'last_step': max(finite_steps) if finite_steps else None,
            'invalid_steps': invalid_steps, 'spike_observations': len(spikes), 'spikes': spikes}


def needs_review(node):
    from loss_curve_health import history_paths
    review = node.get('gradient_curve_review', {})
    return (review.get('policy_version') != POLICY['version']
            or review.get('paths') != history_paths(node) or not review.get('terminal') or review.get('error'))


def review_node(node, api, cache_dir, checked_at, current_run=None):
    from loss_curve_health import history_paths
    paths = history_paths(node)
    terminal = node['status'] in ('succeeded', 'failed', 'canceled')
    current_path = node.get('reported_metrics', {}).get('wandb_path')
    warmup = node.get('summary', {}).get('config', {}).get('training', {}).get('warmup_steps', 192)
    sources = []
    prior = {s['path']: s for s in node.get('gradient_curve_review', {}).get('sources', [])}
    for path in paths:
        cache = Path(cache_dir) / (hashlib.sha256(path.encode()).hexdigest() + '.json')
        immutable = terminal or path != current_path
        cached = json.loads(cache.read_text()) if cache.exists() else {}
        if immutable and cached.get('terminal') and cached.get('path') == path and cached.get('metric') == POLICY['metric']:
            if path in prior and prior[path]['history_sha256'] != hashlib.sha256(cache.read_bytes()).hexdigest():
                raise ValueError('Gradient-history hash mismatch')
        else:
            run = current_run if path == current_path and current_run is not None else api.run(path)
            # Query this metric independently: W&B intersects requested keys, so adding
            # CE here would silently omit steps where only one metric was recorded.
            keys = ['_step', '_timestamp', POLICY['metric']]
            rows = [{k: row.get(k) for k in keys} for row in run.scan_history(keys=keys, page_size=1000)]
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists():
                archive = cache.parent / 'history' / (hashlib.sha256(cache.read_bytes()).hexdigest() + '.json')
                archive.parent.mkdir(parents=True, exist_ok=True)
                if not archive.exists():
                    archive.write_bytes(cache.read_bytes())
            cached = {'path': path, 'url': run.url, 'terminal': immutable, 'metric': POLICY['metric'],
                      'fetched_at': checked_at, 'rows': rows}
            temporary = cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(cached, sort_keys=True) + '\n')
            temporary.replace(cache)
        sources.append({'kind': 'wandb', 'path': path, 'url': cached['url'],
                        'history_file': str(cache), 'history_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
                        **analyze_history(cached['rows'], warmup)})
    status = ('unhealthy' if any(s['status'] == 'unhealthy' for s in sources)
              else 'review_needed' if any(s['status'] == 'review_needed' for s in sources)
              else 'no_spike_detected' if any(s['observations'] for s in sources) else 'unavailable')
    node['gradient_curve_review'] = {'policy_version': POLICY['version'], 'status': status,
        'checked_at': checked_at, 'paths': paths, 'terminal': terminal, 'sources': sources}
