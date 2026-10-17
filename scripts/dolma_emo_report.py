"""Separate, automatically refreshed report for the fixed Dolma3 EMO comparison."""
import copy
import json

from dolma_emo_trials import COHORT, POOLS, VOCABS
from pipeline import EVALS, REPO
from report import build_report_data, build_figures, vocab_label


def comparison_data(state):
    cohort = state['study'][COHORT]
    data = build_report_data(state, report_cohort=COHORT)
    data['pools'] = list(POOLS)
    data['token_vocab_labels'] = [vocab_label(v) for v in VOCABS]
    data['training_tokens'] = [row for row in data['training_tokens']
                               if row['dataset'] == 'Dolma' and row['pool'] in POOLS]
    data['metric_labels'] = {key: data['metric_labels'][key] for key in EVALS}
    data['metric_labels']['dolma_bpb'] = 'Shared validation Dolma BPB ↓'
    data['selected'] = []
    for run in data['runs']:
        root = state['nodes'][run['id']]
        evaluation = state['nodes'][run['id'].replace('train-', 'eval-', 1)]
        summary = evaluation.get('summary', {})
        shared = summary.get('shared_validation') or root.get('shared_validation', {})
        score = shared.get('metrics', {}).get('validation_bpb')
        ready = (run['training_status'] == run['conversion_status'] == run['eval_status'] == 'completed'
                 and run['result_valid'] and not root.get('monitor_error') and not evaluation.get('monitor_error')
                 and set(EVALS) <= summary.get('metrics', {}).keys() and score is not None
                 and all(run['metrics'].get(metric) is not None for metric in EVALS))
        run['evaluation_complete'] = ready
        if ready:
            data['selected'].append(run['id'])
        else:
            # Partial benchmark output and training-time validation are not final endpoints.
            run['metrics'] = {key: None for key in run['metrics']}
        if run['status'] not in ('failed', 'unhealthy', 'canceled'):
            if ready:
                run['status'] = 'completed'
            elif run['training_status'] == 'completed':
                run['status'] = ('converting' if run['conversion_status'] != 'completed' else 'evaluating')
    data['complete'] = len(data['selected']) == len(cohort['runs']) == 36
    data['panels'] = []
    data['figures'] = {metric: [] for metric in EVALS}
    for pool in POOLS:
        for embed_wd in (0., .1):
            view = dict(data, pools=[pool], runs=[r for r in data['runs']
                                                if r['pool'] == pool and r['embedding_wd'] == embed_wd])
            figures = build_figures(view)
            data['panels'].append(dict(pool=pool, embedding_wd=embed_wd))
            for metric in EVALS:
                figure = figures[metric][0]
                if not any(len(trace['x']) for trace in figure['data']):
                    figure['layout']['annotations'][0]['text'] = 'Waiting for completed evaluations'
                data['figures'][metric].append(figure)
    provenance_keys = set(cohort['runs']) | set(cohort['required']) | set(cohort['sources'])
    provenance_keys.update(k.replace('train-', 'convert-', 1) for k in cohort['runs'])
    data['provenance'] = dict(authorization=copy.deepcopy(cohort),
                             nodes={k: copy.deepcopy(state['nodes'][k]) for k in sorted(provenance_keys)})
    return data


def render_report(state, destination):
    from plotly.offline import get_plotlyjs
    from plotly.utils import PlotlyJSONEncoder
    data = comparison_data(state)
    payload = json.dumps(data, cls=PlotlyJSONEncoder, sort_keys=True).replace('<', '\\u003c')
    source = (REPO / 'scripts/dolma_emo_report_template.html').read_text()
    source = source.replace('__REPORT_DATA__', payload).replace('__PLOTLY_JS__', get_plotlyjs())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source)
