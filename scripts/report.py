"""Standalone study report, with measured-token compute and grouped tuning runs."""
from collections import Counter, defaultdict
import copy
from functools import lru_cache
import json
import math
from pathlib import Path

from pipeline import DEFAULT_BATCH_SIZE, EVALS, REPO
from adaptive_study import excluded_run
from loss_curve_health import invalid_result

MODEL_ORDER = ['small', 'medium', 'medium_large']
COLORS = {32000: '#2563eb', 64000: '#8b5cf6', 128000: '#dc681e'}
METRICS = {
    'dolma_bpb': ('Held-out Dolma BPB ↓', False),
    'validation_bpb': ('Validation BPB ↓', False),
    'train_bpb': ('Training BPB ↓', False),
    'blimp': ('BLiMP accuracy ↑', True),
    'arc_easy': ('ARC-Easy accuracy ↑', True),
    'hellaswag': ('HellaSwag accuracy ↑', True),
    'piqa': ('PIQA accuracy ↑', True),
}


def display_status(status):
    return {'pending': 'planned', 'submitted': 'queued', 'initializing': 'running',
            'succeeded': 'completed'}.get(status, status)


def vocab_label(vocab):
    return f'{vocab / 1000:g}K'


def argument(node, name, default=None):
    args = node.get('arguments', [])
    return args[args.index(name) + 1] if name in args else default


def run_config(node):
    saved = node.get('summary', {}).get('config')
    if saved:
        return copy.deepcopy(saved)
    cfg = json.loads((REPO / 'configs' / f"{node['model_size']}.json").read_text())
    for key, cast in [('learning_rate', float), ('weight_decay', float), ('epochs', int),
                      ('seed', int), ('micro_batch_size', int)]:
        cfg['training'][key] = cast(argument(node, '--' + key.replace('_', '-'), cfg['training'][key]))
    cfg['training']['embedding_weight_decay'] = float(argument(node, '--embedding-weight-decay',
        cfg['training'].get('embedding_weight_decay', cfg['training']['weight_decay'])))
    return cfg


@lru_cache(maxsize=128)
def model_compute(model_json, vocab_size, seq_length):
    """Use the exact vendored EMO forward+backward convention on a meta model."""
    from olmo_core.data import TokenizerConfig
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig
    vocab = TokenizerConfig(vocab_size=vocab_size, eos_token_id=0, pad_token_id=0).padded_vocab_size()
    model = TransformerConfig.llama_like(vocab_size=vocab, attn_backend=AttentionBackendName.torch,
                                        **json.loads(model_json)).build(init_device='meta')
    return {'flops_per_token': model.num_flops_per_token(seq_length),
            'parameters': model.num_params, 'padded_vocab_size': vocab}


def compute_budget(node, cfg, meta, baseline):
    batch = int(argument(node, '--batch-size', cfg.get('pipeline', {}).get('global_batch_size', DEFAULT_BATCH_SIZE)))
    result = {'train_tokens': meta.get('train', {}).get('tokens'), 'epoch_tokens': None,
              'epoch_flops': None, 'flops_per_token': None, 'seq_length': None,
              'global_batch_size': batch}
    if not meta or not baseline:
        return result
    for key in ('data_files', 'seed', 'eval_fraction'):
        if meta.get(key) != baseline.get(key):
            raise ValueError(f'FLOPs reference mismatch: {key}')
    for split in ('train', 'eval'):
        for key in ('bytes', 'documents'):
            if meta[split][key] != baseline[split][key]:
                raise ValueError(f'FLOPs reference mismatch: {split}.{key}')
    seq = round(4096 * meta['tokens_per_byte'] / baseline['tokens_per_byte'])
    if node.get('summary', {}).get('config') and seq != cfg['seq_length']:
        raise ValueError('Saved training context differs from measured context')
    tokens_per_step = seq * batch
    epoch_steps = meta['train']['tokens'] // tokens_per_step
    fraction = float(argument(node, '--epoch-fraction', 1))
    steps = int(argument(node, '--max-steps', math.floor(epoch_steps * fraction)))
    epoch_tokens = steps * tokens_per_step
    estimate = model_compute(json.dumps(cfg['model'], sort_keys=True), node['vocab_size'], seq)
    result.update(estimate, seq_length=seq, epoch_tokens=epoch_tokens,
                  epoch_flops=epoch_tokens * estimate['flops_per_token'],
                  dropped_tokens=meta['train']['tokens'] - epoch_tokens,
                  global_batch_size=batch, steps_per_epoch=epoch_tokens // tokens_per_step)
    observed = node.get('summary', {}).get('training', {}).get('flops_per_token')
    if observed is not None and observed != result['flops_per_token']:
        raise ValueError('Report FLOPs differ from the training model')
    return result


def build_report_data(state):
    nodes = state['nodes']
    streams = {(n.get('data_size'), n['vocab_size']): n.get('summary', {}).get('meta', {})
               for n in nodes.values() if n['stage'] == 'tokenize-data' and n['status'] == 'succeeded'}
    counts = []
    for pool in state.get('pool_sizes', ['100GB']):
        ns = [n for n in nodes.values() if n['stage'] == 'train-tokenizer' and n.get('data_size', '100GB') == pool]
        count = Counter(display_status(n['status']) for n in ns)
        counts.append({'pool': pool, 'total': len(ns), **{k: count[k] for k in
                       ('completed', 'running', 'queued', 'planned', 'failed', 'canceled')}})
    runs = []
    for key, node in nodes.items():
        if node['stage'] != 'train-model' or node['status'] == 'canceled':
            continue
        if excluded_run(state, node) or node.get('study_role') == 'superseded_plan' or node.get('legacy_architecture'):
            continue
        pool, vocab, model = node.get('data_size', '100GB'), node['vocab_size'], node['model_size']
        conversion = next((n for n in nodes.values() if n['stage'] == 'convert-model' and key in n['dependencies']), {})
        conversion_key = next((k for k, n in nodes.items() if n is conversion), None)
        evaluation = next((n for n in nodes.values() if n['stage'] == 'evaluate-model' and conversion_key in n['dependencies']), {})
        cfg = run_config(node)
        losses = dict(node.get('reported_metrics', {}))
        losses.update(node.get('summary', {}).get('training', {}).get('latest_metrics', {}))
        metrics = {k: losses.get(k) for k in ('train_bpb', 'validation_bpb')}
        if node.get('shared_validation_required'):
            metrics['validation_bpb'] = node.get('shared_validation', {}).get('metrics', {}).get('validation_bpb')
        for task in EVALS:
            metrics[task] = evaluation.get('summary', {}).get('metrics', {}).get(task, {}).get(
                'bpb' if task == 'dolma_bpb' else 'accuracy')
        if state.get('study', {}).get('dolma_validation_id'):
            # Do not display the old 1000-document score while backfill is pending.
            shared = node.get('shared_validation') or evaluation.get('summary', {}).get('shared_validation', {})
            metrics['dolma_bpb'] = shared.get('metrics', {}).get('validation_bpb')
        budget = compute_budget(node, cfg, streams.get((pool, vocab), {}), streams.get((pool, 32000), {}))
        raw_metrics = dict(metrics)
        invalid = invalid_result(node)
        training_status = display_status(node['status'])
        review_needed = node.get('loss_curve_review', {}).get('status') == 'review_needed'
        if invalid:
            metrics = {name: None for name in metrics}
        runtime_batch = node.get('runtime_batch') or node.get('summary', {}).get('training', {}).get('runtime_batch') or {}
        runs.append({'id': key, 'pool': pool, 'vocab': vocab, 'vocab_label': vocab_label(vocab),
                     'model': model, 'training_status': training_status,
                     'status': 'unhealthy' if invalid else 'review' if review_needed and training_status == 'completed' else training_status,
                     'result_valid': not invalid,
                     'loss_curve_review': node.get('loss_curve_review'),
                     'loss_health_decision': node.get('loss_health_decision'),
                     'eval_status': display_status(evaluation.get('status', 'pending')),
                     'conversion_status': display_status(conversion.get('status', 'pending')),
                     'lr': cfg['training']['learning_rate'], 'wd': cfg['training']['weight_decay'],
                     'batch_size': budget['global_batch_size'],
                     'epoch_fraction': float(argument(node, '--epoch-fraction', 1)),
                     'study_role': node.get('study_role'),
                     'resume_verified': node.get('resume_verified'),
                     'shared_validation': node.get('shared_validation'),
                     'previous_attempts': node.get('previous_attempts', []),
                     'wandb_url': node.get('reported_metrics', {}).get('wandb_url'),
                     'wandb_discovery': node.get('wandb_discovery'),
                     'seed': cfg['training']['seed'], 'epochs': cfg['training']['epochs'],
                     'microbatch': runtime_batch.get('micro_batch_size', cfg['training']['micro_batch_size']),
                     'runtime_batch': runtime_batch or None,
                     'runtime_batch_verified': node.get('runtime_batch_verified', node.get('summary', {}).get('training', {}).get('runtime_batch_verified')),
                     'weight_tying': cfg['model'].get('tie_word_embeddings', False),
                     'embedding_wd': cfg['training'].get('embedding_weight_decay', cfg['training']['weight_decay']),
                     'metrics': metrics, 'raw_metrics': raw_metrics, 'loss_observations': losses, 'compute': budget,
                     'progress': node.get('progress', {}),
                     'error': node.get('monitor_error') or node.get('submission_error'),
                     'url': node.get('receipt', {}).get('url'),
                     'eval_url': evaluation.get('receipt', {}).get('url')})
    groups = defaultdict(list)
    for run in runs:
        groups[(run['pool'], run['vocab'], run['model'], run['epoch_fraction'])].append(run)
    selected = []
    for group in groups.values():
        eligible = [r for r in group if r['training_status'] == 'completed' and r['result_valid']
                    and r['metrics']['validation_bpb'] is not None]
        pool_search = state.get('study', {}).get('pool_lr_search', {})
        fixed_lr = state.get('study', {}).get('fixed_learning_rates', {}).get(group[0]['model'])
        if fixed_lr is not None:
            chosen_lr = fixed_lr
        elif group[0]['pool'] in pool_search.get('pools', []):
            # Compare the authorized rates on this pool's validation split only.
            eligible = [r for r in eligible if r['lr'] in pool_search['learning_rates']]
            chosen_lr = None
        else:
            chosen_lr = state.get('study', {}).get('selected', {}).get(f"{group[0]['model']}:{group[0]['vocab']}", {}).get('lr')
        if chosen_lr is not None:
            eligible = [r for r in eligible if r['lr'] == chosen_lr]
        choice = min(eligible, key=lambda r: (r['metrics']['validation_bpb'], r['id'])) if eligible else group[0] if len(group) == 1 and group[0]['result_valid'] else None
        if choice:
            selected.append(choice['id'])
    pools = state.get('training_pools', ['100GB'])
    runs.sort(key=lambda r: (pools.index(r['pool']), MODEL_ORDER.index(r['model']), r['vocab'], r['lr'], r['wd'], r['batch_size'], r['embedding_wd'], r['id']))
    vocabs = sorted({n['vocab_size'] for n in nodes.values() if n['stage'] == 'tokenize-data'})
    training_tokens = [{'pool': pool, 'tokens': {
        vocab_label(vocab): streams.get((pool, vocab), {}).get('train', {}).get('tokens')
        for vocab in vocabs}} for pool in pools]
    for row in training_tokens:
        baseline = row['tokens'].get('32K')
        row['change_vs_32k_pct'] = {
            label: 100 * (tokens / baseline - 1) if tokens is not None and baseline else None
            for label, tokens in row['tokens'].items()}
    return {'tokenizers': counts, 'runs': runs, 'selected': selected, 'pools': pools,
            'training_tokens': training_tokens, 'token_vocab_labels': [vocab_label(v) for v in vocabs],
            'metric_labels': {k: ('Shared validation Dolma BPB ↓' if k == 'dolma_bpb'
                and state.get('study', {}).get('dolma_validation_id') else v[0])
                for k, v in METRICS.items()
                if k != 'validation_bpb' or not state.get('study', {}).get('dolma_validation_id')},
            'pool_details': [n.get('summary', {}).get('pool', {}) for n in nodes.values() if n['stage'] == 'sample-pool'],
            'complete': state['complete']}


def build_figures(data, horizon=1.):
    import plotly.graph_objects as go
    figures = {}
    for metric, (label, percentage) in METRICS.items():
        label = data['metric_labels'].get(metric, label)
        figures[metric] = []
        for pool in data['pools']:
            fig = go.Figure()
            pool_runs = [r for r in data['runs'] if r['pool'] == pool and r['id'] in data['selected']
                         and r['epoch_fraction'] == (horizon if pool == '100GB' else 1.)]
            plotted = 0
            for vocab in sorted({r['vocab'] for r in pool_runs}):
                records = sorted([r for r in pool_runs if r['vocab'] == vocab and r['training_status'] == 'completed' and r['result_valid']
                                  and r['compute']['epoch_flops'] is not None and r['metrics'][metric] is not None],
                                 key=lambda r: MODEL_ORDER.index(r['model']))
                plotted += len(records)
                fig.add_trace(go.Scatter(
                    x=[r['compute']['epoch_flops'] for r in records],
                    y=[r['metrics'][metric] * (100 if percentage else 1) for r in records],
                    mode='lines+markers', name=vocab_label(vocab),
                    line={'color': COLORS.get(vocab, '#475569'), 'width': 2},
                    marker={'size': 10, 'symbol': [['circle', 'square', 'diamond'][MODEL_ORDER.index(r['model'])] for r in records]},
                    customdata=[[r['model'], r['vocab_label'], r['lr'], r['wd'], r['compute']['epoch_tokens'], r['batch_size'], r['embedding_wd']] for r in records],
                    hovertemplate='%{customdata[0]} · %{customdata[1]}<br>FLOPs %{x:.3e}<br>' + label +
                    ' %{y:.4f}<br>LR %{customdata[2]} · WD %{customdata[3]} · BS %{customdata[5]} · EmbedWD %{customdata[6]}<br>Epoch tokens %{customdata[4]:,}<extra></extra>'))
            measured = [r['compute']['epoch_flops'] for r in pool_runs if r['compute']['epoch_flops']]
            layout = dict(template='plotly_white', height=330, margin=dict(l=62, r=20, t=24, b=65),
                          font=dict(family='Inter, system-ui, sans-serif', size=11, color='#475569'),
                          xaxis=dict(title='One-epoch training FLOPs', type='log', exponentformat='power', gridcolor='#e8edf3'),
                          yaxis=dict(title=label, gridcolor='#e8edf3', zeroline=False),
                          showlegend=False, hovermode='closest', paper_bgcolor='white', plot_bgcolor='white')
            if not plotted:
                message = 'Waiting for tokenization and 32K reference' if not measured else 'FLOPs calculated · waiting for completed-run metrics'
                fig.add_annotation(text=message, x=.5, y=.5, xref='paper', yref='paper', showarrow=False,
                                   font=dict(size=11, color='#64748b'))
                layout['yaxis'].update(showticklabels=False, range=[0, 1])
                if not measured:
                    layout['xaxis'].update(showticklabels=False, range=[17, 20])
                else:
                    layout['xaxis']['range'] = [math.log10(min(measured)) - .15,
                                               math.log10(max(measured)) + .15]
            fig.update_layout(**layout)
            figures[metric].append(fig.to_plotly_json())
    return figures


def render_report(state, destination):
    from plotly.offline import get_plotlyjs
    from plotly.utils import PlotlyJSONEncoder
    evaluations = [n for n in state['nodes'].values() if n['stage'] == 'evaluate-model']
    if study := state.get('study'):
        state['complete'] = study['phase'] == 'complete' and all(
            state['nodes'][k]['status'] == 'succeeded' and set(EVALS) <= set(state['nodes'][k].get('summary', {}).get('metrics', {}))
            and not invalid_result(state['nodes'].get(k.replace('eval-', 'train-', 1), {}))
            for k in study['required'])
    else:
        state['complete'] = bool(evaluations) and all(n['status'] == 'succeeded' for n in state['nodes'].values()) and all(
            set(EVALS) <= set(n.get('summary', {}).get('metrics', {})) for n in evaluations)
    data = build_report_data(state)
    data['figures'] = build_figures(data)
    if state.get('study'):
        data['horizon_figures'] = {f'{h:g}': build_figures(data, h) for h in (.05, .1, .2, .4, 1.)}
        data['study'] = state['study']
    data['campaign'] = state
    payload = json.dumps(data, cls=PlotlyJSONEncoder, sort_keys=True).replace('<', '\\u003c')
    source = (REPO / 'scripts' / 'report_template.html').read_text()
    source = source.replace('__REPORT_DATA__', payload).replace('__PLOTLY_JS__', get_plotlyjs())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
