"""Idempotent LR bracket, transfer, and checkpoint-branch campaign planning."""
import copy
import math
from datetime import datetime, timezone

MODELS = ('small', 'medium', 'medium_large')
HORIZONS = (.05, .1, .2, .4)
PROJECT = 'tokenizer-scaling'


def now():
    return datetime.now(timezone.utc).isoformat()


def arg(node, flag, default=None):
    values = node.get('arguments', [])
    return values[values.index(flag) + 1] if flag in values else default


def completed_score(node):
    if node.get('health_stop') and node['status'] in ('failed', 'canceled'):
        return math.inf
    if node['status'] != 'succeeded':
        return None
    value = node.get('summary', {}).get('training', {}).get('latest_metrics', {}).get('validation_bpb')
    return value if value is not None and math.isfinite(value) else None


def eval_key(train_key):
    return train_key.replace('train-', 'eval-', 1)


def cpu_node(state, key, stage, pool, vocab=None, dependencies=()):
    if key in state['nodes']:
        return key
    args = [stage, '--root', state['study']['root'], '--data-size', pool]
    if stage == 'sample-pool':
        args += ['--sampling-seed', '42']
    else:
        args += ['--vocab-size', str(vocab)]
    state['nodes'][key] = dict(stage=stage, arguments=args, data_size=pool, vocab_size=vocab,
        model_size=None, status='pending', dependencies=list(dependencies), expected_gpus=0,
        min_runtime='3h', placement_clusters=['ai2/saturn', 'ai2/neptune'], allow_gpu_reservation=True)
    return key


def token_stream(state, pool, vocab):
    sample = f'sample-{pool.lower()}'
    if sample not in state['nodes']:
        cpu_node(state, sample, 'sample-pool', pool, dependencies=['download'])
    bpe = cpu_node(state, f'bpe-{pool.lower()}-{vocab}', 'train-tokenizer', pool, vocab, [sample])
    return cpu_node(state, f'tokens-{pool.lower()}-{vocab}', 'tokenize-data', pool, vocab, [bpe])


def model_run(state, pool, model, vocab, lr, *, role, fraction=1., parent=None):
    nodes = state['nodes']
    # Reuse any already-submitted scientifically matching one-epoch run.
    for key, node in nodes.items():
        if (node['stage'] == 'train-model' and node.get('data_size') == pool
                and node.get('model_size') == model and node.get('vocab_size') == vocab
                and float(arg(node, '--learning-rate', 0)) == lr
                and float(arg(node, '--epoch-fraction', 1)) == fraction
                and int(arg(node, '--batch-size', 128)) == 128
                and (node.get('receipt') or not node.get('hold_reason'))):
            node.setdefault('study_role', role)
            return key
    suffix = f'{model}-{pool.lower()}-{vocab}-lr{lr:.8g}'.replace('.', 'p')
    if fraction != 1:
        suffix += f'-ep{fraction:g}'.replace('.', 'p')
    key = 'train-' + suffix
    if key in nodes:
        return key
    deps = sorted({token_stream(state, pool, vocab), token_stream(state, pool, 32000)})
    micro = 4 if model == 'medium_large' and vocab >= 128000 else 8 if model == 'medium_large' else 16
    common = ['--root', state['study']['root'], '--data-size', pool, '--vocab-size', str(vocab),
              '--model-size', model, '--gpus', '8', '--batch-size', '128', '--micro-batch-size', str(micro),
              '--epochs', '1', '--learning-rate', str(lr), '--epoch-fraction', str(fraction),
              '--wandb-project', PROJECT, '--health-check', '--eval-interval',
              '5000' if pool == '100GB' else '250' if pool == '5GB' else '1000']
    if pool == '100GB' and fraction == 1:
        common += ['--decay-horizons', *map(str, HORIZONS)]
    if parent:
        record = nodes[parent].get('branch_checkpoints', {}).get(f'{fraction:g}')
        if not record:
            raise ValueError('Cannot branch before the full-state checkpoint is published')
        common += ['--branch-from', record['parent']]
    for prefix, stage in [('train', 'train-model'), ('convert', 'convert-model'), ('eval', 'evaluate-model')]:
        child = f'{prefix}-{suffix}'
        nodes[child] = dict(stage=stage, arguments=[stage, *common], status='pending',
            dependencies=deps, model_size=model, vocab_size=vocab, data_size=pool,
            expected_gpus=8 if prefix == 'train' else int(prefix == 'eval'),
            min_runtime='3h', priority='urgent', placement_clusters=['ai2/jupiter'],
            study_role=role, epoch_fraction=fraction, study_parent=parent)
        if prefix == 'train':
            nodes[child].update(chain_evaluation=True, wandb_secret='SEWONM_WANDB_API_KEY')
        deps = [child]
    return key


def initialize(state, gpu_scope='all_user'):
    if state.get('study'):
        return
    state['study'] = dict(version=1, started_at=now(), root='/oe-training-default/sewonm/tokenizer_project',
        gpu_limit=100, gpu_scope=gpu_scope, selection_metric='validation_bpb',
        brackets={}, selected={}, required=[], phase='lr_tuning', cohorts=[[32000, 64000, 128000]],
        smaller_vocab_decision=None, monitor_minutes=30)
    for key, node in state['nodes'].items():
        if node['stage'] == 'train-model':
            node['study_role'] = 'baseline'
            if node['status'] == 'pending' and not node.get('receipt'):
                node['hold_reason'] = 'Superseded pending recipe; await LR chosen on 5GB'
                node['study_role'] = 'superseded_plan'
            elif node.get('data_size') == '5GB':
                pair = f"{node['model_size']}:{node['vocab_size']}"
                state['study']['brackets'][pair] = {'baseline': key}
        if not node.get('receipt'):
            node['min_runtime'] = '3h'
    state.pop('submission_hold', None)


def trend_decision(state):
    """A transparent descriptive trigger, not a significance test."""
    rows = []
    for pool in ('5GB', '10GB', '20GB'):
        for model in MODELS:
            values = {}
            for vocab in (32000, 64000, 128000):
                key = state['study']['transfers'][f'{pool}:{model}:{vocab}'] if pool != '5GB' else state['study']['selected'][f'{model}:{vocab}']['key']
                metric = state['nodes'][eval_key(key)].get('summary', {}).get('metrics', {}).get('dolma_bpb', {}).get('bpb')
                if metric is None:
                    return None
                values[vocab] = metric
            rows.append({'pool': pool, 'model': model, 'bpb': values,
                         'small_vocab_better': values[32000] + .001 < min(values[64000], values[128000])})
    wins = sum(r['small_vocab_better'] for r in rows)
    small_pool_wins = sum(r['small_vocab_better'] for r in rows if r['pool'] == '5GB')
    return {'triggered': wins >= 5 and small_pool_wins >= 2, 'rows': rows,
            'rule': '32K BPB at least 0.001 below both 64K and 128K in >=5/9 small-pool coordinates, including >=2/3 on 5GB',
            'decided_at': now()}


def advance_plan(state):
    study = state.get('study')
    if not study:
        return
    nodes = state['nodes']
    vocabs = study['cohorts'][-1]
    required = set(study['required'])
    all_selected = True
    for model in MODELS:
        for vocab in vocabs:
            pair = f'{model}:{vocab}'
            bracket = study['brackets'].setdefault(pair, {})
            if 'baseline' not in bracket:
                lr = study['selected'][f'{model}:32000']['lr']
                bracket['baseline'] = model_run(state, '5GB', model, vocab, lr, role='lr_baseline')
            base = nodes[bracket['baseline']]
            lr = float(arg(base, '--learning-rate'))
            bracket.setdefault('lower', model_run(state, '5GB', model, vocab, lr / 2, role='lr_trial'))
            base_score, lower_score = completed_score(base), completed_score(nodes[bracket['lower']])
            if base_score is not None and lower_score is not None:
                if 'next' not in bracket:
                    next_lr = lr / 4 if lower_score < base_score else lr * 2
                    bracket['direction'] = 'lower_again' if lower_score < base_score else 'higher'
                    bracket['next'] = model_run(state, '5GB', model, vocab, next_lr, role='lr_trial')
                scores = [(completed_score(nodes[bracket[k]]), bracket[k]) for k in ('baseline', 'lower', 'next')]
                if all(score is not None for score, _ in scores):
                    score, winner = min(scores, key=lambda x: (x[0], x[1]))
                    if math.isfinite(score):
                        study['selected'][pair] = dict(key=winner, lr=float(arg(nodes[winner], '--learning-rate')),
                            validation_bpb=score, candidates=[key for _, key in scores])
            for kind in ('baseline', 'lower', 'next'):
                if kind in bracket and not nodes[bracket[kind]].get('health_stop'):
                    required.add(eval_key(bracket[kind]))
                elif kind in bracket:
                    required.discard(eval_key(bracket[kind]))
            if pair not in study['selected']:
                all_selected = False
    study['required'] = sorted(required)
    if not all_selected:
        study['phase'] = 'lr_tuning'
        return
    transfers = study.setdefault('transfers', {})
    for pool in ('10GB', '20GB'):
        for model in MODELS:
            for vocab in vocabs:
                pair = f'{model}:{vocab}'
                key = model_run(state, pool, model, vocab, study['selected'][pair]['lr'], role='selected_lr')
                transfers[f'{pool}:{pair}'] = key
                required.add(eval_key(key))
    study['required'] = sorted(required)
    if any(nodes[eval_key(transfers[f'{pool}:{m}:{v}'])]['status'] != 'succeeded'
           for pool in ('10GB', '20GB') for m in MODELS for v in vocabs):
        study['phase'] = 'transfer_10gb_20gb'
        return
    for field in ('pool_sizes', 'training_pools'):
        if '100GB' not in state[field]:
            state[field].append('100GB')
    hundred = study.setdefault('hundred_gb', {})
    for model in MODELS:
        for vocab in vocabs:
            pair = f'{model}:{vocab}';lr = study['selected'][pair]['lr']
            trunk = model_run(state, '100GB', model, vocab, lr, role='100gb_trunk')
            hundred[pair] = trunk
            required.add(eval_key(trunk))
            for fraction in HORIZONS:
                if f'{fraction:g}' in nodes[trunk].get('branch_checkpoints', {}):
                    key = model_run(state, '100GB', model, vocab, lr, role='100gb_decay_branch', fraction=fraction, parent=trunk)
                    required.add(eval_key(key))
    study['required'] = sorted(required)
    trunks_done = all(nodes[eval_key(hundred[f'{m}:{v}'])]['status'] == 'succeeded' for m in MODELS for v in vocabs)
    if not trunks_done or any(nodes[k]['status'] != 'succeeded' for k in required):
        study['phase'] = '100gb_horizons'
        return
    if study['smaller_vocab_decision'] is None:
        study['smaller_vocab_decision'] = trend_decision(state)
        if study['smaller_vocab_decision']['triggered']:
            study['cohorts'].append([8000, 16000])
            study['phase'] = 'smaller_vocab_lr_tuning'
            advance_plan(state)
            return
    study['phase'] = 'complete'


def reservation(node):
    if node.get('expected_gpus', 0):
        return node['expected_gpus']
    # Weka GPU clusters may reserve a complete eight-GPU node for a CPU request.
    return 0 if node.get('placement_clusters') == ['ai2/phobos'] else 8


def live_gpu_budget(state):
    from beaker import Beaker
    from google.protobuf.json_format import MessageToDict
    with Beaker.from_env(default_workspace=state['workspace'], check_for_upgrades=False) as b:
        author = b.user.get(state['owner'])
        own = {n['receipt']['workload_id'] for n in state['nodes'].values() if n.get('receipt')}
        entries = []
        for w in b.workload.list(author=author, finalized=False):
            identifier = w.experiment.id or w.environment.id
            if state['study']['gpu_scope'] == 'study' and identifier not in own:
                continue
            # Enumerate all live jobs/replicas, not only a workload's latest job.
            full = b.workload.get(identifier)
            jobs = ([job for task in full.experiment.tasks for job in b.job.list(task=task, finalized=False)]
                    if full.experiment.id else list(b.job.list(environment=full.environment, finalized=False)))
            if not jobs and full.experiment.id:
                spec = b.experiment.get_spec(full).to_json()
                reserve = sum(max(8, int(t.get('resources', {}).get('gpuCount', 0))) * int(t.get('replicas', 1))
                              for t in spec['tasks'])
                entries.append({'job': None, 'workload': identifier, 'requested': reserve,
                                'assigned': 0, 'reserved': reserve})
            for job in jobs:
                    d = MessageToDict(job)
                    requested = int(d.get('containerSpec', {}).get('resourceRequest', {}).get('gpuCount', 0))
                    assigned = len(d.get('assignmentDetails', {}).get('resourceAssignment', {}).get('gpus', []))
                    reserve = max(requested, assigned)
                    if reserve == 0:
                        # Conservative even for an unassigned external CPU job.
                        reserve = 8
                    entries.append({'job': job.id, 'workload': identifier, 'requested': requested,
                                    'assigned': assigned, 'reserved': reserve})
        observed = {e['workload'] for e in entries}
        for node in state['nodes'].values():
            if node.get('chain_parent') or not node.get('receipt'):
                continue
            identifier = node['receipt']['workload_id']
            status = node.get('chain_job_status', node['status'])
            if identifier not in observed and status not in ('succeeded', 'failed', 'canceled'):
                entries.append({'job': None, 'workload': identifier, 'requested': reservation(node),
                                'assigned': 0, 'reserved': reservation(node)})
                observed.add(identifier)
        return {'checked_at': now(), 'entries': entries,
                'assigned': sum(e['assigned'] for e in entries),
                'reserved': sum(e['reserved'] for e in entries), 'limit': state['study']['gpu_limit']}


def monitor_wandb(state):
    """Read the actual W&B runs as well as retaining the in-job loss guard."""
    targets = [n for n in state['nodes'].values() if n['stage'] == 'train-model'
               and n.get('reported_metrics', {}).get('wandb_path')
               and (n['status'] not in ('succeeded', 'failed', 'canceled') or not n.get('wandb_final_checked'))]
    if not targets:
        return
    try:
        from beaker import Beaker
        import wandb
        with Beaker.from_env(default_workspace=state['workspace'], check_for_upgrades=False) as b:
            ws = b.workspace.get(state['workspace'])
            key = b.secret.read(b.secret.get('SEWONM_WANDB_API_KEY', workspace=ws), workspace=ws)
        api = wandb.Api(api_key=key, timeout=20)
        del key
        for node in targets:
            run = api.run(node['reported_metrics']['wandb_path'])
            metrics = dict(run.summary)
            node['wandb_observation'] = {'checked_at': now(), 'state': run.state, 'url': run.url,
                'metrics': {k: v for k, v in metrics.items() if k in
                    ('train/CE loss', 'eval/lm/validation/CE loss', 'optim/grad_norm', '_step')
                    and isinstance(v, (int, float)) and math.isfinite(v)}}
            if node['status'] in ('succeeded', 'failed', 'canceled'):
                node['wandb_final_checked'] = True
        state.pop('wandb_monitor_error', None)
    except Exception as error:
        # Do not serialize exception text: auth failures may include sensitive request details.
        state['wandb_monitor_error'] = type(error).__name__ + ': W&B query failed; Beaker loss logs and in-job health guard remain available'
