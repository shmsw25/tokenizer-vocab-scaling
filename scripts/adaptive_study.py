"""Idempotent LR bracket, transfer, and checkpoint-branch campaign planning."""
import copy
import math
from datetime import datetime, timezone
from loss_curve_health import invalid_result

MODELS = ('small', 'medium', 'medium_large')
HORIZONS = (.05, .1, .2, .4)
PROJECT = 'tokenizer-scaling'


def now():
    return datetime.now(timezone.utc).isoformat()


def arg(node, flag, default=None):
    values = node.get('arguments', [])
    return values[values.index(flag) + 1] if flag in values else default


def learning_rate_allowed(state, lr):
    return float(lr) not in state.get('study', {}).get('excluded_learning_rates', [])


def excluded_run(state, node):
    lr = arg(node, '--learning-rate', node.get('summary', {}).get('config', {}).get('training', {}).get('learning_rate'))
    return bool(node.get('lr_excluded') or (lr is not None and not learning_rate_allowed(state, lr)))


def completed_score(node):
    # A suspected infrastructure failure is not evidence against a learning rate.
    if node.get('failure_investigation', {}).get('status') == 'unresolved':
        return None
    if invalid_result(node):
        return math.inf
    if node.get('health_stop') and node['status'] in ('failed', 'canceled'):
        return math.inf
    if node['status'] != 'succeeded':
        return None
    if node.get('shared_validation_required'):
        value = node.get('shared_validation', {}).get('metrics', {}).get('validation_bpb')
    else:
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
        min_runtime=state['study'].get('pool_min_runtime', {}).get(pool, '3h'),
        placement_clusters=['ai2/saturn', 'ai2/neptune'], allow_gpu_reservation=True)
    return key


def token_stream(state, pool, vocab):
    sample = f'sample-{pool.lower()}'
    if sample not in state['nodes']:
        cpu_node(state, sample, 'sample-pool', pool, dependencies=['download'])
    bpe = cpu_node(state, f'bpe-{pool.lower()}-{vocab}', 'train-tokenizer', pool, vocab, [sample])
    return cpu_node(state, f'tokens-{pool.lower()}-{vocab}', 'tokenize-data', pool, vocab, [bpe])


def model_run(state, pool, model, vocab, lr, *, role, fraction=1., parent=None,
              weight_decay=.1, batch_size=128, embedding_weight_decay=None):
    from runtime_batch import GPU_CLUSTERS
    nodes = state['nodes']
    embedding_wd = weight_decay if embedding_weight_decay is None else embedding_weight_decay
    if not learning_rate_allowed(state, lr):
        raise ValueError(f'Learning rate {lr:g} was excluded by the user')
    fixed_lr = state['study'].get('fixed_learning_rates', {}).get(model)
    additional = state['study'].get('additional_learning_rates', {}).get(pool, {}).get(model, [])
    if fixed_lr is not None and lr != fixed_lr and lr not in additional:
        raise ValueError(f'{model} uses the user-fixed learning rate {fixed_lr}')
    # Reuse any already-submitted scientifically matching one-epoch run.
    for key, node in nodes.items():
        if (node['stage'] == 'train-model' and not excluded_run(state, node) and not node.get('legacy_architecture') and node.get('data_size') == pool
                and node.get('model_size') == model and node.get('vocab_size') == vocab
                and float(arg(node, '--learning-rate', 0)) == lr
                and float(arg(node, '--epoch-fraction', 1)) == fraction
                and int(arg(node, '--batch-size', 128)) == batch_size
                and float(arg(node, '--weight-decay', .1)) == weight_decay
                and float(arg(node, '--embedding-weight-decay', arg(node, '--weight-decay', .1))) == embedding_wd
                and (node.get('receipt') or not node.get('hold_reason'))):
            node.setdefault('study_role', role)
            return key
    suffix = f'{model}-{pool.lower()}-{vocab}-lr{lr:.8g}'.replace('.', 'p')
    if state['study'].get('architecture_revision') and model == 'medium_large':
        suffix += '-prenorm-v2'
    if fraction != 1:
        suffix += f'-ep{fraction:g}'.replace('.', 'p')
    if weight_decay != .1 or batch_size != 128 or embedding_wd != weight_decay:
        suffix += f'-wd{weight_decay:g}-bs{batch_size}-embedwd{embedding_wd:g}'.replace('.', 'p')
    key = 'train-' + suffix
    if key in nodes:
        return key
    deps = sorted({token_stream(state, pool, vocab), token_stream(state, pool, 32000)})
    if state['study'].get('validation_node'):
        deps.append(state['study']['validation_node'])
        if pool not in ('5GB', '10GB', '20GB') or vocab not in (32000, 64000, 128000):
            deps.append(cpu_node(state, f'validation-tokens-{pool.lower()}-{vocab}',
                'tokenize-validation', pool, vocab, [token_stream(state, pool, vocab), state['study']['validation_node']]))
    if model == 'medium_large' and state['study'].get('architecture_archive_node'):
        deps.append(state['study']['architecture_archive_node'])
    micro = 4 if model == 'medium_large' and vocab >= 128000 else 8 if model == 'medium_large' else 16
    common = ['--root', state['study']['root'], '--data-size', pool, '--vocab-size', str(vocab),
              '--model-size', model, '--gpus', '8', '--batch-size', str(batch_size), '--micro-batch-size', str(micro),
              '--epochs', '1', '--learning-rate', str(lr), '--epoch-fraction', str(fraction),
              '--wandb-project', PROJECT, '--health-check', '--eval-interval',
              '5000' if pool == '100GB' else '250' if pool == '5GB' else '1000']
    if weight_decay != .1 or embedding_weight_decay is not None:
        common += ['--weight-decay', str(weight_decay)]
    if embedding_wd != weight_decay:
        common += ['--embedding-weight-decay', str(embedding_wd)]
    if state['study'].get('validation_node'):
        common += ['--validation-id', 'dolma100m-v1']
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
            min_runtime=state['study'].get('pool_min_runtime', {}).get(pool, '3h'),
            priority='urgent', placement_clusters=['ai2/jupiter'],
            study_role=role, epoch_fraction=fraction, study_parent=parent)
        if prefix == 'train':
            nodes[child].update(chain_evaluation=True, wandb_secret='SEWONM_WANDB_API_KEY',
                placement_clusters=list(state['study'].get('training_placement_clusters', GPU_CLUSTERS)))
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
                if invalid_result(state['nodes'][key]):
                    return None
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


def advance_pool_lr_search(state, vocabs, required):
    """Run both authorized rates per larger-pool coordinate, reusing matching runs."""
    study, nodes = state['study'], state['nodes']
    search = study['pool_lr_search']
    ready = True
    for pool in search['pools']:
        for model in MODELS:
            for vocab in vocabs:
                coordinate = f'{pool}:{model}:{vocab}'
                rates = ([study['fixed_learning_rates'][model]]
                         if model in study.get('fixed_learning_rates', {}) else search['learning_rates'])
                keys = [model_run(state, pool, model, vocab, lr, role='pool_lr_trial')
                        for lr in rates if learning_rate_allowed(state, lr)]
                if not keys:
                    ready = False
                    continue
                search.setdefault('runs', {})[coordinate] = keys
                for key in keys:
                    if invalid_result(nodes[key]) or (nodes[key].get('health_stop') and nodes[key]['status'] in ('failed', 'canceled')):
                        required.discard(eval_key(key))
                    else:
                        required.add(eval_key(key))
                        if nodes[eval_key(key)]['status'] != 'succeeded':
                            ready = False
                scores = [(completed_score(nodes[key]), key) for key in keys]
                if all(score is not None for score, _ in scores):
                    score, winner = min(scores, key=lambda x: (x[0], x[1]))
                    if math.isfinite(score):
                        search.setdefault('selected', {})[coordinate] = dict(
                            key=winner, lr=float(arg(nodes[winner], '--learning-rate')),
                            validation_bpb=score, candidates=keys)
                        study.setdefault('transfers', {})[coordinate] = winner
                    else:
                        ready = False
                else:
                    ready = False
    return ready


def select_explicit_five_gb_trials(state, pair, required):
    """Evaluate only the user's recorded candidates, without expanding their grid."""
    study, nodes = state['study'], state['nodes']
    keys = [key for key in study['five_gb_lr_search']['runs'][pair]
            if not excluded_run(state, nodes[key])]
    ready = bool(keys)
    scores = []
    for key in keys:
        node = nodes[key]
        score = completed_score(node)
        if invalid_result(node) or (node.get('health_stop') and node['status'] in ('failed', 'canceled')):
            required.discard(eval_key(key))
        else:
            required.add(eval_key(key))
            ready &= nodes[eval_key(key)]['status'] == 'succeeded'
        ready &= score is not None
        if score is not None and math.isfinite(score):
            scores.append((score, key))
    if scores:
        score, winner = min(scores)
        study['selected'][pair] = dict(key=winner, lr=float(arg(nodes[winner], '--learning-rate')),
            validation_bpb=score, candidates=keys, selection='explicit_5gb_hyperparameters',
            embedding_weight_decay=float(arg(nodes[winner], '--embedding-weight-decay', .1)))
    else:
        study['selected'].pop(pair, None)
    return ready and bool(scores)


def advance_plan(state):
    study = state.get('study')
    if not study:
        return
    # Invalidation must also revoke cached winners and transfer decisions.
    for choices in (study.get('selected', {}), study.get('pool_lr_search', {}).get('selected', {})):
        for coordinate, choice in list(choices.items()):
            if invalid_result(state['nodes'][choice['key']]) or excluded_run(state, state['nodes'][choice['key']]):
                del choices[coordinate]
    for coordinate, key in list(study.get('transfers', {}).items()):
        if invalid_result(state['nodes'][key]) or excluded_run(state, state['nodes'][key]):
            del study['transfers'][coordinate]
    sync_shared_validation(state)
    nodes = state['nodes']
    vocabs = study['cohorts'][-1]
    required = {key for key in study['required'] if not excluded_run(state, nodes[key])}
    all_selected = True
    for model in MODELS:
        for vocab in vocabs:
            pair = f'{model}:{vocab}'
            if pair in study.get('five_gb_lr_search', {}).get('runs', {}):
                all_selected &= select_explicit_five_gb_trials(state, pair, required)
                continue
            fixed_lr = study.get('fixed_learning_rates', {}).get(model)
            if fixed_lr is not None:
                key = model_run(state, '5GB', model, vocab, fixed_lr, role='fixed_lr')
                study['brackets'][pair] = {'baseline': key, 'fixed_lr': fixed_lr}
                score = completed_score(nodes[key])
                if not invalid_result(nodes[key]):
                    study['selected'][pair] = dict(key=key, lr=fixed_lr, validation_bpb=score,
                        candidates=[key], selection='fixed_by_user')
                if invalid_result(nodes[key]):
                    required.discard(eval_key(key))
                else:
                    required.add(eval_key(key))
                if score is None or not math.isfinite(score): all_selected = False
                continue
            bracket = study['brackets'].setdefault(pair, {})
            if 'next' in bracket and excluded_run(state, nodes[bracket['next']]):
                bracket['skipped_next_lr'] = float(arg(nodes[bracket.pop('next')], '--learning-rate'))
            if pair in study['selected'] and excluded_run(state, nodes[study['selected'][pair]['key']]):
                del study['selected'][pair]
            if 'baseline' not in bracket:
                lr = study['selected'].get(f'{model}:32000', {}).get('lr',
                    .001 if model == 'medium_large' else .002)
                bracket['baseline'] = model_run(state, '5GB', model, vocab, lr, role='lr_baseline')
            base = nodes[bracket['baseline']]
            lr = float(arg(base, '--learning-rate'))
            bracket.setdefault('lower', model_run(state, '5GB', model, vocab, lr / 2, role='lr_trial'))
            base_score, lower_score = completed_score(base), completed_score(nodes[bracket['lower']])
            if base_score is not None and lower_score is not None:
                if 'next' not in bracket and 'skipped_next_lr' not in bracket:
                    next_lr = lr / 4 if lower_score < base_score else lr * 2
                    bracket['direction'] = 'lower_again' if lower_score < base_score else 'higher'
                    if learning_rate_allowed(state, next_lr):
                        bracket['next'] = model_run(state, '5GB', model, vocab, next_lr, role='lr_trial')
                    else:
                        bracket['skipped_next_lr'] = next_lr
                scores = [(completed_score(nodes[bracket[k]]), bracket[k])
                          for k in ('baseline', 'lower', 'next') if k in bracket]
                if all(score is not None for score, _ in scores):
                    score, winner = min(scores, key=lambda x: (x[0], x[1]))
                    if math.isfinite(score):
                        study['selected'][pair] = dict(key=winner, lr=float(arg(nodes[winner], '--learning-rate')),
                            validation_bpb=score, candidates=[key for _, key in scores])
            for kind in ('baseline', 'lower', 'next'):
                if kind in bracket and not nodes[bracket[kind]].get('health_stop') and not invalid_result(nodes[bracket[kind]]):
                    required.add(eval_key(bracket[kind]))
                elif kind in bracket:
                    required.discard(eval_key(bracket[kind]))
            if (pair not in study['selected'] or ('next' not in bracket and 'skipped_next_lr' not in bracket)
                    or any(completed_score(nodes[bracket[kind]]) is None
                           for kind in ('baseline', 'lower', 'next') if kind in bracket)):
                all_selected = False
    if study.get('pool_lr_search'):
        search_ready = advance_pool_lr_search(state, vocabs, required)
        study['required'] = sorted(required)
        if not search_ready or not all_selected:
            study['phase'] = 'pool_lr_search'
            return
    study['required'] = sorted(required)
    if not all_selected:
        study['phase'] = 'lr_tuning'
        return
    transfers = study.setdefault('transfers', {})
    for pool in (() if study.get('pool_lr_search') else ('10GB', '20GB')):
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
            pair = f'{model}:{vocab}'
            lr = study.get('fixed_learning_rates', {}).get(model, study['selected'][pair]['lr'])
            trunk = model_run(state, '100GB', model, vocab, lr, role='100gb_trunk')
            hundred[pair] = trunk
            required.add(eval_key(trunk))
            for fraction in HORIZONS:
                if f'{fraction:g}' in nodes[trunk].get('branch_checkpoints', {}):
                    key = model_run(state, '100GB', model, vocab, lr, role='100gb_decay_branch', fraction=fraction, parent=trunk)
                    required.add(eval_key(key))
    study['required'] = sorted(required)
    trunks_done = all(nodes[eval_key(hundred[f'{m}:{v}'])]['status'] == 'succeeded'
                      and not invalid_result(nodes[hundred[f'{m}:{v}']]) for m in MODELS for v in vocabs)
    branches_published = all({f'{h:g}' for h in HORIZONS} <= set(nodes[hundred[f'{m}:{v}']].get('branch_checkpoints', {}))
                             for m in MODELS for v in vocabs)
    if not trunks_done or not branches_published or any(nodes[k]['status'] != 'succeeded'
            or invalid_result(nodes.get(k.replace('eval-', 'train-', 1), {})) for k in required):
        study['phase'] = '100gb_horizons'
        return
    if study['smaller_vocab_decision'] is None:
        study['smaller_vocab_decision'] = trend_decision(state)
        if study['smaller_vocab_decision']['triggered']:
            study['cohorts'].append([8000, 16000])
            study['phase'] = 'smaller_vocab_lr_tuning'
            advance_plan(state)
            return
    if any(nodes[k]['status'] != 'succeeded' for k in study.get('shared_validation_nodes', [])):
        study['phase'] = 'shared_validation'
        return
    study['phase'] = 'complete'


def reservation(node):
    if node.get('expected_gpus', 0):
        return node['expected_gpus']
    # Weka GPU clusters may reserve a complete eight-GPU node for a CPU request.
    return 0 if node.get('placement_clusters') == ['ai2/phobos'] else 8


def live_gpu_budget(state):
    from beaker import Beaker, beaker_pb2 as pb
    from beaker._service_client import RpcMethod
    from google.protobuf.json_format import MessageToDict
    import yaml
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
                # Read the server spec directly: beaker-py 2.5 cannot decode
                # context.minRuntime, which our queued jobs explicitly set.
                # A failed read still propagates to the fail-closed admission check.
                raw = b.experiment.rpc_request(
                    RpcMethod[pb.GetExperimentYamlSpecResponse](b.experiment.service.GetExperimentYamlSpec),
                    pb.GetExperimentYamlSpecRequest(experiment_id=full.experiment.id),
                ).experiment_spec
                spec = yaml.safe_load(raw)
                if not isinstance(spec, dict) or not spec.get('tasks'):
                    raise ValueError(f'Missing tasks in queued workload spec: {identifier}')
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
    from loss_curve_health import discover_beaker_history, history_paths, needs_review, review_node
    from pipeline import REPO
    cache_dir = REPO / '.build/loss-curve-health/histories'
    candidates = [n for n in state['nodes'].values() if n['stage'] == 'train-model'
                  and not excluded_run(state, n) and not n.get('legacy_architecture')
                  and n.get('study_role') != 'superseded_plan'
                  and (n['status'] != 'canceled' or n.get('unhealthy_stop'))]
    missing = [n for n in candidates if not n.get('reported_metrics', {}).get('wandb_path') and n.get('receipt')
               and n['status'] in ('running', 'succeeded', 'failed')]
    if missing:
        try:
            from beaker import Beaker
            with Beaker.from_env(default_workspace=state['workspace'], check_for_upgrades=False) as b:
                for node in missing:
                    try:
                        discover_beaker_history(node, b, cache_dir / 'beaker', now())
                        if not history_paths(node) and needs_review(node):
                            review_node(node, None, cache_dir, now())
                        node.pop('loss_history_discovery_error', None)
                    except Exception as error:
                        node['loss_history_discovery_error'] = type(error).__name__
        except Exception as error:
            state['loss_history_discovery_error'] = type(error).__name__ + ': Beaker history discovery failed'
        else:
            state.pop('loss_history_discovery_error', None)
    targets = [n for n in candidates if history_paths(n)
               and (n['status'] not in ('succeeded', 'failed', 'canceled')
                    or not n.get('wandb_final_checked') or needs_review(n))]
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
        failed = []
        for node in targets:
            try:
                path = node.get('reported_metrics', {}).get('wandb_path')
                run = api.run(path) if path else None
                if run is not None:
                    node.setdefault('reported_metrics', {})['wandb_url'] = run.url
                    metrics = dict(run.summary)
                    node['wandb_observation'] = {'checked_at': now(), 'state': run.state, 'url': run.url,
                        'metrics': {k: v for k, v in metrics.items() if k in
                            ('train/CE loss', 'eval/lm/validation/CE loss', 'optim/grad_norm', '_step')
                            and isinstance(v, (int, float)) and math.isfinite(v)}}
                review_node(node, api, cache_dir, now(), current_run=run)
                if node['status'] in ('succeeded', 'failed', 'canceled'):
                    node['wandb_final_checked'] = True
            except Exception as error:
                # Keep previous evidence and human decisions; auth details are never serialized.
                node.setdefault('loss_curve_review', {})['error'] = type(error).__name__ + ': loss history query failed'
                failed.append(type(error).__name__)
        state.pop('wandb_monitor_error', None)
        if failed:
            state['wandb_monitor_error'] = f'W&B history unavailable for {len(failed)} run(s); prior observations preserved'
    except Exception as error:
        # Do not serialize exception text: auth failures may include sensitive request details.
        state['wandb_monitor_error'] = type(error).__name__ + ': W&B query failed; Beaker loss logs and in-job health guard remain available'


def sync_shared_validation(state):
    """Rescore old and new completed models before comparing validation losses."""
    from submit import VALIDATION_CLUSTERS
    study=state['study']
    if not study.get('validation_node'):return
    nodes=state['nodes']; required=[]
    for key,n in list(nodes.items()):
        if n['stage']!='train-model' or excluded_run(state,n) or n.get('legacy_architecture') or n.get('study_role')=='superseded_plan' or invalid_result(n):continue
        n['shared_validation_required']=True
        vk=key.replace('train-','validation-',1)
        evaluation=nodes.get(eval_key(key),{})
        shared = evaluation.get('summary', {}).get('shared_validation')
        if shared and evaluation.get('status') == 'succeeded':
            n['shared_validation'] = shared
            continue  # New shells already score the shared corpus in their evaluation stage.
        if vk in nodes:
            required.append(vk)
            if nodes[vk]['status']=='succeeded':
                n['shared_validation']=nodes[vk]['summary']
                if study.get('dolma_validation_id'):
                    from score_validation import dolma_metric
                    evaluation['summary']['metrics']['dolma_bpb'] = dolma_metric(n['shared_validation'])
                    evaluation['summary']['shared_validation'] = n['shared_validation']
            continue
        if n['status']!='succeeded' or evaluation.get('status')!='succeeded':continue
        directory=n.get('summary',{}).get('model_dir')
        if not directory:continue
        nodes[vk]=dict(stage='validate-model',arguments=['validate-model','--root',study['root'],
            '--data-size',n['data_size'],'--vocab-size',str(n['vocab_size']),'--model-directory',directory],
            data_size=n['data_size'],vocab_size=n['vocab_size'],model_size=n['model_size'],
            status='pending',dependencies=[eval_key(key),study['validation_node']],expected_gpus=1,
            min_runtime='30m',priority='urgent',
            placement_clusters=list(study.get('validation_placement_clusters', VALIDATION_CLUSTERS)),
            validation_parent=key)
        required.append(vk)
    study['shared_validation_nodes']=required
    backfill = study.get('validation_backfill')
    if backfill:
        backfill['completed'] = [key for key in backfill['targets'] if nodes[key].get('shared_validation')]
        backfill['skipped_unhealthy'] = [key for key in backfill['targets']
            if invalid_result(nodes[key])
            or (nodes[key].get('health_stop') and nodes[key]['status'] in ('failed', 'canceled'))]
        backfill['skipped_excluded'] = [key for key in backfill['targets'] if excluded_run(state,nodes[key])]
        if len(set(backfill['completed'] + backfill['skipped_unhealthy'] + backfill['skipped_excluded'])) == len(backfill['targets']):
            backfill['status'] = 'complete'
            # This file is only the one-time local migration/launch script, never a worker.
            from pathlib import Path
            from pipeline import REPO, digest
            script = REPO / '.build/validation-replacement/migrate.py'
            if script.exists() and digest(script) == backfill.get('script_sha256'):
                script.unlink()
                backfill['script_deleted_at'] = now()
