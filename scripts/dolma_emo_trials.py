"""Explicit Dolma3 / EMO / RepFilter comparison, isolated from the legacy study."""
import argparse
import copy
import fcntl
import hashlib
import json
import shutil

from campaign import now, prepare_chains
from emo_trials import digest
from pipeline import REPO, read_json, write_json
from repetition_filter import FILTER_NAME, FILTER_SETTINGS
from runtime_batch import GPU_CLUSTERS
from training_recipe import EMO_RECIPE

COHORT = 'dolma_emo_repfilter_v1'
REPORT_NAME = 'dolma-emo-report.html'
AUDIT = REPO / '.build/dolma-emo-repfilter-20261017'
POOLS = ('5GB', '10GB')
MODELS = ('small', 'medium', 'medium_large')
VOCABS = (32000, 64000, 128000)
DEFAULTS = dict(dataset='dolma', training_recipe=EMO_RECIPE,
                repetition_filter=FILTER_NAME, learning_rate=0.0005,
                maximum_learning_rate=0.0005, weight_decay=0.1,
                batch_size=128, epochs=1, seed=42, validation_id='dolma100m-v1')


def authorize_runs(state):
    """Add exactly 36 fresh roots; preserve every existing node and hold."""
    study, nodes = state['study'], state['nodes']
    if study.get(COHORT):
        raise ValueError('Dolma EMO comparison already authorized; do not duplicate it')
    sources, additions, roots, required = {}, {}, [], []

    def verified(key):
        node = nodes[key]
        if (node['status'] != 'succeeded' or node.get('monitor_error')
                or node['summary']['stage'] != node['stage']
                or node['summary']['revision'] != node['receipt']['revision']):
            raise ValueError(f'Prerequisite source identity is unverified: {key}')
        sources[key] = dict(receipt=copy.deepcopy(node['receipt']),
                            summary_sha256=digest(node['summary']))
        return node['summary']

    validation_key = study['validation_node']
    preparation = verified(validation_key)
    corpus = preparation['validation']
    for pool in POOLS:
        baseline = verified(f'tokens-{pool.lower()}-32000')['meta']
        for vocab in VOCABS:
            meta = verified(f'tokens-{pool.lower()}-{vocab}')['meta']
            if (meta['vocab_size'] != vocab or meta['seed'] != 42 or meta['eval_fraction'] != .01
                    or meta['data_files'] != baseline['data_files']
                    or any(meta[s][k] != baseline[s][k]
                           for s in ('train', 'eval') for k in ('bytes', 'documents'))):
                raise ValueError(f'Training stream identity mismatch: {pool}/{vocab}')
            stream = preparation['streams'][f'default_{pool}_{vocab}']
            recipe = stream['recipe']
            if (recipe['validation_id'] != 'dolma100m-v1' or recipe['pool'] != pool
                    or recipe['vocab_size'] != vocab or recipe.get('tokenizer_dataset', 'dolma') != 'dolma'
                    or recipe['documents_sha256'] != corpus['sha256']
                    or stream['documents'] != corpus['documents']
                    or stream['bytes'] != corpus['text_bytes']):
                raise ValueError(f'Shared validation identity mismatch: {pool}/{vocab}')
    archive_key = study.get('architecture_archive_node')
    if archive_key:
        archive = nodes[archive_key]
        info = archive.get('summary', {})
        if (archive['status'] != 'succeeded' or archive.get('monitor_error')
                or archive.get('job_status', {}).get('status') != 'STATUS_SUCCEEDED'
                or info.get('source') != study['root'] + '/models/medium_large'
                or info.get('target') != study['root'] + '/models/medium_large_old'
                or not info.get('config_files_checked') or not info.get('run_directories')):
            raise ValueError('Corrected architecture archive prerequisite is unverified')
        sources[archive_key] = dict(receipt=copy.deepcopy(archive['receipt']), summary_sha256=digest(info))
    for model in MODELS:
        config = read_json(REPO / 'configs' / f'{model}.json')['model']
        if config.get('block_name') != 'default' or config.get('qk_norm') is not False:
            raise ValueError(f'Expected pre-norm/no-QK architecture: {model}')
    # Admit all model sizes and both pools early, including the largest memory case.
    for vocab in (128000, 32000, 64000):
        for model in MODELS:
            for pool in POOLS:
                baseline_key = f'tokens-{pool.lower()}-32000'
                stream_key = f'tokens-{pool.lower()}-{vocab}'
                meta = nodes[stream_key]['summary']['meta']
                baseline = nodes[baseline_key]['summary']['meta']
                micro = 4 if model == 'medium_large' and vocab == 128000 else 8 if model == 'medium_large' else 16
                for embed_wd in (0., .1):
                    common = ['--root', study['root'], '--dataset', 'dolma', '--data-size', pool,
                        '--vocab-size', str(vocab), '--model-size', model, '--gpus', '8',
                        '--batch-size', '128', '--micro-batch-size', str(micro), '--learning-rate', '0.0005',
                        '--weight-decay', '0.1', '--embedding-weight-decay', f'{embed_wd:g}',
                        '--epochs', '1', '--epoch-fraction', '1', '--seed', '42',
                        '--validation-id', 'dolma100m-v1', '--wandb-project', 'tokenizer-scaling',
                        '--eval-interval', '1000', '--training-recipe', EMO_RECIPE,
                        '--repetition-filter', FILTER_NAME]
                    suffix = f'dolma-{model}-{pool.lower()}-{vocab}-lr0p0005-emo-dense-v1-repfilter-embedwd{embed_wd:g}'.replace('.', 'p')
                    dependencies = sorted({stream_key, baseline_key, validation_key}
                                          | ({archive_key} if archive_key and model == 'medium_large' else set()))
                    for prefix, stage in [('train', 'train-model'), ('convert', 'convert-model'), ('eval', 'evaluate-model')]:
                        key = f'{prefix}-{suffix}'
                        if key in nodes:
                            raise ValueError(f'Comparison node already exists: {key}')
                        additions[key] = dict(stage=stage, dataset='dolma', arguments=[stage, *common],
                            status='pending', dependencies=dependencies, data_size=pool, vocab_size=vocab,
                            model_size=model, epoch_fraction=1., expected_gpus=8 if prefix == 'train' else int(prefix == 'eval'),
                            min_runtime='1h' if pool == '5GB' else '3h', priority='urgent',
                            study_role='explicit_dolma_emo_repfilter', report_cohort=COHORT,
                            placement_clusters=list(GPU_CLUSTERS) if prefix == 'train' else ['ai2/jupiter'],
                            validation_stream_node=validation_key)
                        if prefix == 'train':
                            seq = round(4096 * meta['tokens_per_byte'] / baseline['tokens_per_byte'])
                            additions[key].update(chain_evaluation=True, wandb_secret='SEWONM_WANDB_API_KEY',
                                admission_order=len(roots),
                                planned_tokens=dict(train_tokens=meta['train']['tokens'], seq_length=seq,
                                    epoch_steps=meta['train']['tokens'] // (seq * 128)),
                                validation_provenance=copy.deepcopy(preparation['streams'][f'default_{pool}_{vocab}']))
                            roots.append(key)
                        if prefix == 'eval':
                            required.append(key)
                        dependencies = [key]
    authorized_at = now()
    # Build chain links on the additions only, without rewriting historical nodes.
    prepare_chains({'nodes': additions})
    nodes.update(additions)
    for key in roots:
        study.setdefault('training_submission_exceptions', {})[key] = dict(
            source='explicit_user_request', authorized_at=authorized_at,
            arguments_sha256=hashlib.sha256(json.dumps(nodes[key]['arguments'], separators=(',', ':')).encode()).hexdigest(),
            reason='36 fresh Dolma3 EMO RepFilter runs: 5GB/10GB x small/medium/medium_large x 32K/64K/128K x EmbedWD0/0.1, LR5e-4')
    study['experiment_defaults'] = copy.deepcopy(DEFAULTS)
    study['maximum_new_learning_rate'] = .0005
    study[COHORT] = dict(authorized_at=authorized_at, runs=roots, required=required, sources=sources,
        pools=list(POOLS), models=list(MODELS), vocabularies=list(VOCABS), embedding_weight_decays=[0., .1],
        defaults=copy.deepcopy(DEFAULTS), repetition_filter=copy.deepcopy(FILTER_SETTINGS), report=REPORT_NAME,
        scope='Exactly 36 fresh runs with same-allocation chained evaluations; lower LR may be considered later, never higher; no automatic additional trials')
    study['required'] = sorted(set(study['required']) | set(required))
    return study[COHORT]


def sources_match(state):
    """Recheck frozen prerequisites at admission, including after a controller restart."""
    cohort = state.get('study', {}).get(COHORT)
    if not cohort:
        return False
    for key, frozen in cohort['sources'].items():
        node = state['nodes'].get(key, {})
        if (node.get('status') != 'succeeded' or node.get('monitor_error')
                or node.get('receipt') != frozen['receipt']
                or digest(node.get('summary')) != frozen['summary_sha256']):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-36-runs', action='store_true', required=True)
    parser.parse_args()
    path = REPO / '.build/campaign.json'
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (AUDIT / 'state-before.json').exists():
            raise ValueError('Authorization backup exists; inspect rather than rerun')
        original = read_json(path)
        state = copy.deepcopy(original)
        authorization = authorize_runs(state)
        assert all(state['nodes'][k] == n for k, n in original['nodes'].items())
        write_json(AUDIT / 'state-before.json', original)
        if path.with_name('report.html').exists():
            shutil.copy2(path.with_name('report.html'), AUDIT / 'original-report.html')
        write_json(AUDIT / 'authorization.json', authorization)
        write_json(path, state)
        print(json.dumps(authorization, indent=2))


if __name__ == '__main__':
    main()
