"""Explicit admission of the six requested small/10GB EMO-recipe comparisons."""
import argparse
import copy
import fcntl
import hashlib
import json

from campaign import now, prepare_chains
from pipeline import REPO, read_json, write_json
from runtime_batch import GPU_CLUSTERS
from training_recipe import EMO_RECIPE

COHORT = 'emo_small_10gb'
AUDIT = REPO / '.build/emo-small-10gb-20261016'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def authorize_six_runs(state):
    """Extend in place without changing any prior node, receipt or hold."""
    study, nodes = state['study'], state['nodes']
    if study.get(COHORT):
        raise ValueError('Six EMO trials already authorized; do not duplicate them')
    additions, roots, required, sources = {}, [], [], {}

    def verified(key):
        node = nodes[key]
        if (node['status'] != 'succeeded' or node.get('monitor_error')
                or node['summary']['stage'] != node['stage']
                or node['summary']['revision'] != node['receipt']['revision']):
            raise ValueError(f'Prerequisite source identity is unverified: {key}')
        sources[key] = {'receipt': copy.deepcopy(node['receipt']),
                        'summary_sha256': digest(node['summary'])}
        return node['summary']

    preparation = verified(study['validation_node'])
    corpus = preparation['validation']
    for dataset in ('dolma', 'dclm'):
        prefix = 'dclm-' if dataset == 'dclm' else ''
        reference_key = f'{prefix}tokens-10gb-32000'
        baseline = verified(reference_key)['meta']
        for vocab in (32000, 64000, 128000):
            stream_key = f'{prefix}tokens-10gb-{vocab}'
            meta = verified(stream_key)['meta']
            if (meta['vocab_size'] != vocab or meta['seed'] != 42 or meta['eval_fraction'] != .01
                    or meta['data_files'] != baseline['data_files']
                    or any(meta[s][k] != baseline[s][k]
                           for s in ('train', 'eval') for k in ('bytes', 'documents'))):
                raise ValueError(f'Training stream identity mismatch: {stream_key}')
            validation_key = (f'validation-tokens-dclm-10gb-{vocab}' if dataset == 'dclm'
                              else study['validation_node'])
            validation = (verified(validation_key)['validation'] if dataset == 'dclm'
                          else preparation['streams'][f'default_10GB_{vocab}'])
            recipe = validation['recipe']
            if (recipe['validation_id'] != 'dolma100m-v1' or recipe['pool'] != '10GB'
                    or recipe['vocab_size'] != vocab or recipe.get('tokenizer_dataset', 'dolma') != dataset
                    or recipe['documents_sha256'] != corpus['sha256']
                    or validation['documents'] != corpus['documents']
                    or validation['bytes'] != corpus['text_bytes']):
                raise ValueError(f'Shared validation identity mismatch: {dataset}/{vocab}')
            common = ['--root', study['root'], '--dataset', dataset, '--data-size', '10GB',
                '--vocab-size', str(vocab), '--model-size', 'small', '--gpus', '8',
                '--batch-size', '128', '--micro-batch-size', '16', '--learning-rate', '0.0005',
                '--weight-decay', '0.1', '--embedding-weight-decay', '0.1', '--epochs', '1',
                '--epoch-fraction', '1', '--seed', '42', '--validation-id', 'dolma100m-v1',
                '--wandb-project', 'tokenizer-scaling', '--eval-interval', '1000',
                '--training-recipe', EMO_RECIPE]
            suffix = f'{dataset}-small-10gb-{vocab}-lr0p0005-emo-dense-v1'
            dependencies = sorted({stream_key, reference_key, validation_key})
            for stage_prefix, stage in [('train', 'train-model'), ('convert', 'convert-model'), ('eval', 'evaluate-model')]:
                key = f'{stage_prefix}-{suffix}'
                if key in nodes:
                    raise ValueError(f'EMO comparison node already exists: {key}')
                additions[key] = dict(stage=stage, dataset=dataset, arguments=[stage, *common],
                    status='pending', dependencies=dependencies, data_size='10GB', vocab_size=vocab,
                    model_size='small', epoch_fraction=1., expected_gpus=8 if stage_prefix == 'train' else int(stage_prefix == 'eval'),
                    min_runtime='1h', priority='urgent', study_role='explicit_emo_comparison',
                    placement_clusters=list(GPU_CLUSTERS) if stage_prefix == 'train' else ['ai2/jupiter'],
                    validation_stream_node=validation_key)
                if stage_prefix == 'train':
                    seq = round(4096 * meta['tokens_per_byte'] / baseline['tokens_per_byte'])
                    additions[key].update(chain_evaluation=True, wandb_secret='SEWONM_WANDB_API_KEY',
                        planned_tokens={'train_tokens': meta['train']['tokens'], 'seq_length': seq,
                                        'epoch_steps': meta['train']['tokens'] // (seq * 128)},
                        validation_provenance=copy.deepcopy(validation))
                    roots.append(key)
                if stage_prefix == 'eval':
                    required.append(key)
                dependencies = [key]
    authorized_at = now()
    nodes.update(additions)
    for key in roots:
        study.setdefault('training_submission_exceptions', {})[key] = dict(
            source='explicit_user_request', authorized_at=authorized_at,
            arguments_sha256=hashlib.sha256(json.dumps(nodes[key]['arguments'], separators=(',', ':')).encode()).hexdigest(),
            reason='Six fresh small/10GB EMO-recipe runs: Dolma and DCLM x 32K/64K/128K; LR5e-4 WD0.1 BS128 EmbedWD0.1; minRuntime1h')
    study[COHORT] = dict(authorized_at=authorized_at, runs=roots, required=required, sources=sources,
        training_recipe=EMO_RECIPE, min_runtime='1h',
        scope='Exactly six fresh runs with chained evaluations; no automatic LR descent or replacement runs')
    study['required'] = sorted(set(study['required']) | set(required))
    prepare_chains(state)
    return study[COHORT]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-six-runs', action='store_true', required=True)
    parser.parse_args()
    path = REPO / '.build/campaign.json'
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (AUDIT / 'state-before.json').exists():
            raise ValueError('Authorization backup exists; inspect rather than rerun')
        state = read_json(path)
        original = copy.deepcopy(state)
        authorization = authorize_six_runs(state)
        if any(state['nodes'][k] != n for k, n in original['nodes'].items()):
            raise ValueError('Existing nodes changed during authorization')
        write_json(AUDIT / 'state-before.json', original)
        write_json(AUDIT / 'authorization.json', authorization)
        write_json(path, state)
        print(json.dumps(authorization, indent=2))


if __name__ == '__main__':
    main()
