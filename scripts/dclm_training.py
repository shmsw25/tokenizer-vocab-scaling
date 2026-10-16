"""Explicit, one-time admission of the three user-requested DCLM 10GB trials."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path

from campaign import now, prepare_chains
from pipeline import REPO, read_json, write_json
from runtime_batch import GPU_CLUSTERS


def authorize_small_10gb(state, dclm):
    from report import dclm_training_tokens
    study = state['study']
    if study.get('dclm_small_10gb'):
        raise ValueError('DCLM model trials already authorized; do not duplicate the extension')
    rows = dclm_training_tokens(dclm)
    row = next(r for r in rows if r['pool'] == '10GB')
    if not all(row['tokens'].values()) or dclm.get('tokenization_verification_error'):
        raise ValueError('All three verified DCLM 10GB token streams are required')
    nodes = state['nodes']
    roots, required, sources = [], [], {}
    for vocab in (32000, 64000, 128000):
        source_key = f'dclm-tokens-10gb-{vocab}'
        source = dclm['nodes'][source_key]
        if (source['status'] != 'succeeded' or source.get('monitor_error')
                or source['summary']['stage'] != 'tokenize-dclm-data'
                or source['summary']['revision'] != source['receipt']['revision']):
            raise ValueError('DCLM token stream source identity is unverified')
        if source_key in nodes:
            raise ValueError(f'DCLM source already imported: {source_key}')
        nodes[source_key] = copy.deepcopy(source)
        nodes[source_key].update(dataset='dclm', external_campaign='dclm-preparation', dependencies=[])
        sources[source_key] = {'receipt': copy.deepcopy(source['receipt']),
            'summary_sha256': hashlib.sha256(json.dumps(source['summary'], sort_keys=True).encode()).hexdigest()}
        validation = f'validation-tokens-dclm-10gb-{vocab}'
        nodes[validation] = dict(stage='tokenize-validation', dataset='dclm', status='pending',
            data_size='10GB', vocab_size=vocab, model_size=None, expected_gpus=0,
            dependencies=[source_key, study['validation_node']], priority='normal', min_runtime='3h',
            placement_clusters=['ai2/saturn', 'ai2/neptune'], allow_gpu_reservation=True,
            arguments=['tokenize-validation', '--root', study['root'], '--dataset', 'dclm',
                       '--data-size', '10GB', '--vocab-size', str(vocab)])
        suffix = f'dclm-small-10gb-{vocab}-lr0p0005'
        common = ['--root', study['root'], '--dataset', 'dclm', '--data-size', '10GB',
            '--vocab-size', str(vocab), '--model-size', 'small', '--gpus', '8', '--batch-size', '128',
            '--micro-batch-size', '16', '--learning-rate', '0.0005', '--weight-decay', '0.1',
            '--embedding-weight-decay', '0.1', '--epochs', '1', '--epoch-fraction', '1', '--seed', '42',
            '--validation-id', 'dolma100m-v1', '--wandb-project', 'tokenizer-scaling',
            '--health-check', '--eval-interval', '1000']
        dependencies = sorted({source_key, 'dclm-tokens-10gb-32000', validation})
        for prefix, stage in [('train', 'train-model'), ('convert', 'convert-model'), ('eval', 'evaluate-model')]:
            key = f'{prefix}-{suffix}'
            if key in nodes: raise ValueError(f'DCLM model node already exists: {key}')
            nodes[key] = dict(stage=stage, dataset='dclm', arguments=[stage, *common],
                status='pending', dependencies=dependencies, data_size='10GB', vocab_size=vocab,
                model_size='small', expected_gpus=8 if prefix == 'train' else int(prefix == 'eval'),
                min_runtime='3h', priority='urgent', study_role='dclm_explicit_trial',
                placement_clusters=list(GPU_CLUSTERS) if prefix == 'train' else ['ai2/jupiter'],
                validation_stream_node=validation)
            if prefix == 'train':
                nodes[key].update(chain_evaluation=True, wandb_secret='SEWONM_WANDB_API_KEY')
                roots.append(key)
                study.setdefault('training_submission_exceptions', {})[key] = dict(
                    source='explicit_user_request', authorized_at=now(),
                    arguments_sha256=hashlib.sha256(json.dumps(nodes[key]['arguments'], separators=(',', ':')).encode()).hexdigest(),
                    reason='Three fresh small/DCLM/10GB trials at LR5e-4, WD0.1, BS128, EmbedWD0.1')
            if prefix == 'eval': required.append(key)
            dependencies = [key]
    study['dclm_small_10gb'] = dict(authorized_at=now(), runs=roots, required=required, sources=sources,
        validation='dolma100m-v1 encoded with the DCLM tokenizers',
        scope='Exactly three fresh runs; no automatic LR descent or other DCLM model experiments')
    study['required'] = sorted(set(study['required']) | set(required))
    prepare_chains(state)
    return study['dclm_small_10gb']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-small-10gb', action='store_true', required=True)
    args = parser.parse_args()
    del args
    path = REPO / '.build/campaign.json'
    audit = REPO / '.build/dclm-small-10gb-20261015'
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(path)
        if (audit / 'state-before.json').exists():
            raise ValueError('Authorization backup exists; inspect rather than rerun migration')
        original = copy.deepcopy(state)
        authorization = authorize_small_10gb(state, read_json(REPO / '.build/dclm-preparation/campaign.json'))
        write_json(audit / 'state-before.json', original)
        write_json(audit / 'authorization.json', authorization)
        write_json(path, state)
        print(json.dumps(authorization, indent=2))


if __name__ == '__main__':
    main()
