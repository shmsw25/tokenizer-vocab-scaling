"""Train a dense Llama-like language model on data tokenized by tokenize_data.py.

    torchrun --standalone --nproc_per_node=2 train.py \
        --config configs/small.json --data-dir tokenized/32k --output-dir runs/small_32k

Model dimensions and experiment hyperparameters come from the config JSON.
The versioned training recipe supplies optimization and precision settings;
fresh runs default to EMO's dense recipe. The vocabulary size, EOS token and
tokenizer come from the tokenized data directory.

Sequence length, in tokens, is chosen in this order:
    --seq-length N                 use exactly N
    --match-context-of OTHER_DIR   scale the config's seq_length so one sequence
                                   covers the same number of bytes as it does
                                   for the tokenizer in OTHER_DIR
    otherwise                      the config's seq_length

Training resumes automatically when --output-dir already holds checkpoints.
Uses EMO's OLMo-core training stack; checkpoints are in OLMo-core format.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.utils import get_rank
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelWrappingStrategy
from olmo_core.optim import AdamWConfig, SkipStepAdamWConfig, OptimGroupOverride, WSD
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import Callback, ConfigSaverCallback, LMEvaluatorCallbackConfig, WandBCallback
from olmo_core.train.train_module import TransformerDataParallelConfig, TransformerTrainModuleConfig
from scripts.training_callbacks import (LossHealthCallback, ReportMetricsCallback,
                                        RuntimeBatchEvidenceCallback, WSDCheckpointerCallback)
from scripts.training_recipe import (EMO_RECIPE, LEGACY_RECIPE, RECIPES, EMO_SETTINGS,
                                     configure_recipe, checkpoint_recipe_guard)
from scripts.repetition_filter import (FILTER_CHOICES, build_instance_filter,
                                       configure_repetition_filter, verify_instance_filter)


def load_meta(data_dir):
    meta_path = Path(data_dir) / 'meta.json'
    if not meta_path.is_file():
        raise FileNotFoundError(f'{meta_path} is missing; run tokenize_data.py first')
    return json.loads(meta_path.read_text())


def optimizer_config(training, *, recipe=EMO_RECIPE, fused=True):
    """Override only the embedding matrix; a tied output head shares this parameter."""
    wd = training['weight_decay']
    embedding_wd = training.get('embedding_weight_decay', wd)
    if any(not math.isfinite(value) or value < 0 for value in (wd, embedding_wd)):
        raise ValueError('Weight decay and embedding weight decay must be finite and nonnegative')
    overrides = ([OptimGroupOverride(params=['embeddings.weight'],
                                    opts={'weight_decay': embedding_wd})]
                 if embedding_wd != wd else None)
    if recipe == LEGACY_RECIPE:
        return AdamWConfig(lr=training['learning_rate'], weight_decay=wd,
                           group_overrides=overrides, fused=fused)
    if recipe != EMO_RECIPE:
        raise ValueError(f'Unknown training recipe: {recipe}')
    settings = EMO_SETTINGS['optimizer']
    return SkipStepAdamWConfig(
        lr=training['learning_rate'], weight_decay=wd, group_overrides=overrides,
        betas=tuple(settings['betas']), eps=settings['eps'], foreach=settings['foreach'],
        step_increment_bugfix=settings['step_increment_bugfix'],
        rolling_interval_length=settings['rolling_interval_length'], sigma_factor=settings['sigma_factor'])


def train_module_config(config, world_size, *, recipe=EMO_RECIPE):
    """The production entry point and regression tests share this exact builder."""
    configure_recipe(config, recipe)
    training, seq_length = config['training'], config['seq_length']
    emo = recipe == EMO_RECIPE
    parallel = (TransformerDataParallelConfig(
        name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32,
        wrapping_strategy=TransformerDataParallelWrappingStrategy.full) if emo else
        TransformerDataParallelConfig(name=DataParallelType.ddp) if world_size > 1 else None)
    return TransformerTrainModuleConfig(
        dp_config=parallel,
        rank_microbatch_size=training['micro_batch_size'] * seq_length,
        max_sequence_length=seq_length, optim=optimizer_config(training, recipe=recipe),
        scheduler=WSD(warmup=training['warmup_steps'], decay_fraction=training['decay_fraction'],
                      decay_min_lr=training['decay_min_lr']),
        z_loss_multiplier=EMO_SETTINGS['z_loss_multiplier'] if emo else None,
        max_grad_norm=1.0, compile_model=training['compile'],
        autocast_precision=None if emo else DType.bfloat16)


def verify_live_recipe(module, config, *, recipe=EMO_RECIPE):
    """Check instantiated/restored objects, not just requested JSON fields."""
    from olmo_core.optim.adamw import SkipStepAdamW
    from torch.distributed.fsdp import FSDPModule
    def require(condition, message):
        if not condition:
            raise RuntimeError('Training recipe mismatch: ' + message)

    if recipe == EMO_RECIPE:
        require(isinstance(module.optim, SkipStepAdamW), 'EMO requires SkipStepAdamW')
        require(isinstance(module.model, FSDPModule), 'EMO requires actual FSDP wrapping')
        require(module.dp_config is not None and module.dp_config.name == DataParallelType.fsdp,
                'data parallelism must be FSDP')
        require(module.dp_config.param_dtype == DType.bfloat16
                and module.dp_config.reduce_dtype == DType.float32
                and module.dp_config.wrapping_strategy == TransformerDataParallelWrappingStrategy.full,
                'FSDP precision/wrapping configuration changed')
        # Inspect the policy installed on every actual FSDP group as well as its config.
        for wrapped in module.model.modules():
            if isinstance(wrapped, FSDPModule):
                policy = wrapped._get_fsdp_state()._mp_policy
                require(policy.param_dtype == torch.bfloat16 and policy.reduce_dtype == torch.float32,
                        'actual FSDP mixed precision policy changed')
        require(module.autocast_precision is None, 'EMO uses FSDP casting, not autocast')
        require(module.z_loss_multiplier == EMO_SETTINGS['z_loss_multiplier'], 'z-loss changed')
        require(module.optim.foreach and module.optim.stepfix, 'optimizer update implementation changed')
        require(module.optim.dtype is None and all(p.dtype == torch.float32 for p in module.model.parameters()),
                'EMO requires FP32 master parameters and optimizer moments')
        require(module.optim.rolling_interval_length == 128 and module.optim.sigma_factor == 6,
                'outlier handling changed')
        require(module.model.init_tied_output, 'tied initialization order changed')
        require(torch.get_float32_matmul_precision() == 'highest', 'FP32 matmul policy changed')
        require(module.max_grad_norm == 1.0, 'gradient clipping changed')
        betas = (.9, .95)
    else:
        require(recipe == LEGACY_RECIPE and isinstance(module.optim, torch.optim.AdamW),
                'legacy requires AdamW')
        betas = (.9, .999)
    for group in module.optim.param_groups:
        if tuple(group['betas']) != betas or group['eps'] != 1e-8:
            raise RuntimeError('Restored optimizer hyperparameters differ from the training recipe')
    if config['model'].get('tie_word_embeddings'):
        require(module.model.embeddings.weight is module.model.lm_head.w_out.weight,
                'embedding and output weights are no longer tied')
    return verify_optimizer_groups(module.model, module.optim, config['training'])


class TrainingRecipeCallback(Callback):
    def __init__(self, config, model_config, module_config, output_dir, recipe, dataset=None):
        self.config, self.model_config, self.module_config = config, model_config, module_config
        self.output_dir, self.recipe = output_dir, recipe
        self.dataset = dataset

    def pre_train(self):
        # Trainer.fit() may auto-restore: verify after that, before any updates.
        groups = verify_live_recipe(self.trainer.train_module, self.config, recipe=self.recipe)
        filter_record = verify_instance_filter(self.dataset, self.config) if self.dataset is not None else None
        tokenizer_record = None
        if source := self.config.get('pipeline', {}).get('tokenizer_source'):
            if (self.dataset is None or self.dataset.vocab_size != 100278
                    or self.dataset.eos_token_id != 100257 or self.dataset.pad_token_id != 100277
                    or self.model_config.vocab_size != 100352):
                raise RuntimeError('Actual Dolma2 tokenizer/model token IDs differ from EMO')
            tokenizer_record = {'source': source, 'vocab_size': self.dataset.vocab_size,
                'eos_token_id': self.dataset.eos_token_id, 'pad_token_id': self.dataset.pad_token_id,
                'padded_vocab_size': self.model_config.vocab_size}
        if get_rank() == 0:
            from scripts.pipeline import write_json
            module = self.trainer.train_module
            record = {'recipe': self.recipe, 'model': self.model_config.as_config_dict(),
                      'train_module': self.module_config.as_config_dict(),
                      'optimizer_groups': groups,
                      'actual_optimizer': type(module.optim).__name__,
                      'actual_model_wrapper': type(module.model).__name__,
                      'actual_betas': [list(g['betas']) for g in module.optim.param_groups],
                      'actual_eps': [g['eps'] for g in module.optim.param_groups],
                      'actual_z_loss_multiplier': module.z_loss_multiplier,
                      'actual_autocast_precision': str(module.autocast_precision),
                      'actual_callbacks': sorted(self.trainer.callbacks),
                      'float32_matmul_precision': torch.get_float32_matmul_precision()}
            if filter_record is not None:
                record['repetition_filter'] = filter_record
                print('REPETITION_FILTER_VERIFIED ' + json.dumps(filter_record), flush=True)
            if tokenizer_record is not None:
                record['tokenizer_source'] = tokenizer_record
                print('TOKENIZER_SOURCE_VERIFIED ' + json.dumps(tokenizer_record), flush=True)
            write_json(self.output_dir / 'resolved_training_config.json', record)
            print('TRAINING_RECIPE_VERIFIED ' + json.dumps(record), flush=True)


def verify_optimizer_groups(model, optimizer, training):
    """Check actual optimizer membership before training and retain compact evidence."""
    embedding = model.embeddings.weight
    wd = training['weight_decay']
    embedding_wd = training.get('embedding_weight_decay', wd)
    seen = set()
    evidence = []
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if id(parameter) in seen:
                raise RuntimeError('Parameter appears in multiple optimizer groups')
            seen.add(id(parameter))
            expected = embedding_wd if parameter is embedding else wd
            if group['weight_decay'] != expected:
                raise RuntimeError('Optimizer weight decay differs from the requested recipe')
        evidence.append({'weight_decay': group['weight_decay'],
                         'parameter_tensors': len(group['params']),
                         'parameters': sum(p.numel() for p in group['params']),
                         'includes_embedding': any(p is embedding for p in group['params'])})
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Optimizer does not cover every trainable parameter exactly once')
    return evidence


def training_dataset(data_dir, meta, *, data_loader='repack', repetition_filter='none', **dataset_kwargs):
    """Default to repack without changing epoch 1 or existing output fingerprints."""
    if data_loader not in ('repack', 'fsl'):
        raise ValueError(f'Unknown data loader: {data_loader}')
    repack_kwargs = {}
    if data_loader == 'repack':
        sidecar = Path(data_dir) / 'repack.json'
        provenance = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        repack_kwargs = dict(
            dynamic_repacking=True,
            document_metadata_path=str(Path(data_dir) / 'train.documents.csv.gz'),
            num_documents=meta['train']['documents'],
            document_metadata_digest=provenance.get('train', {}).get('sha256'))
    return NumpyFSLDatasetConfig(
        paths=[str(Path(data_dir) / 'train.bin')],
        instance_filter_config=build_instance_filter(repetition_filter),
        **dataset_kwargs, **repack_kwargs).build()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--training-recipe', choices=RECIPES,
                        help='Fresh runs default to emo-dense-v1; legacy is explicit recovery only')
    parser.add_argument('--repetition-filter', choices=FILTER_CHOICES,
                        help='Default: saved config, otherwise none; emo-v1 masks repetitive training sequences')
    parser.add_argument('--data-dir', type=Path, required=True, help='Output directory of tokenize_data.py')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seq-length', type=int)
    parser.add_argument('--match-context-of', type=Path, metavar='OTHER_DIR',
                        help='Tokenized directory of the reference tokenizer')
    parser.add_argument('--micro-batch-size', type=int, help='Sequences per GPU per forward pass')
    parser.add_argument('--gradient-accumulation-steps', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--max-steps', type=int, help='Stop after this many steps instead of the configured epochs')
    parser.add_argument('--data-loader', choices=['repack', 'fsl'],
                        help='Default: training.data_loader in config, otherwise repack')
    parser.add_argument('--run-name', help='W&B run name; defaults to the output directory name')
    parser.add_argument('--resume-from', type=Path, help='Require a full-state checkpoint in this output directory')
    parser.add_argument('--runtime-batch-plan', type=Path, help='Verified GPU-dependent microbatch plan from the pre-launch script')
    cli = parser.parse_args()

    cfg = json.loads(cli.config.read_text())
    recipe = cli.training_recipe or cfg.get('training_recipe', {}).get('name', EMO_RECIPE)
    configure_recipe(cfg, recipe)
    repetition_filter = configure_repetition_filter(cfg, cli.repetition_filter)
    if cli.runtime_batch_plan:
        from scripts.runtime_batch import apply_plan, cuda_device
        if cli.micro_batch_size is not None or cli.gradient_accumulation_steps is not None:
            raise ValueError('Cannot combine a runtime batch plan with manual batching overrides')
        apply_plan(cfg, json.loads(cli.runtime_batch_plan.read_text()),
                   int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('LOCAL_RANK', '0')),
                   cuda_device(torch, int(os.environ.get('LOCAL_RANK', '0'))))
    t = cfg['training']
    t['data_loader'] = cli.data_loader or t.get('data_loader', 'repack')
    for key in ('micro_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'max_steps'):
        if getattr(cli, key) is not None:
            t[key] = getattr(cli, key)
    data_dir, output_dir = cli.data_dir.resolve(), cli.output_dir.resolve()
    # Guard explicit and automatic restores before GPU/model allocation.
    for checkpoint in output_dir.glob('step[0-9]*'):
        if checkpoint.is_dir():
            checkpoint_recipe_guard(checkpoint, cfg)
    if cli.resume_from:
        checkpoint_recipe_guard(cli.resume_from, cfg)
    if t.get('branch_from'):
        boundary = t['max_steps'] - round(t['max_steps'] * t['decay_fraction'])
        checkpoint_recipe_guard(Path(t['branch_from']) / f'step{boundary}', cfg)
    meta = load_meta(data_dir)

    if cli.seq_length is not None:
        cfg['seq_length'] = cli.seq_length
    elif cli.match_context_of is not None:
        # Same bytes per sequence: fewer tokens per byte means a shorter sequence.
        reference = load_meta(cli.match_context_of)
        cfg['seq_length'] = round(cfg['seq_length'] * meta['tokens_per_byte'] / reference['tokens_per_byte'])
    seq_length = cfg['seq_length']
    if seq_length < 2:
        raise ValueError('seq_length must be at least 2')

    # Recorded in the checkpoint config for conversion and evaluation.
    cfg['eos_token'] = meta['eos_token']
    if 'pad_token' in meta:
        cfg['pad_token'] = meta['pad_token']
    cfg['tokenized_dir'] = str(data_dir)
    cfg['tokens_per_byte'] = meta['tokens_per_byte']
    if len(meta['data_files']) == 1:
        cfg['data_file'] = meta['data_files'][0]

    # EMO requires an environment key even when W&B has a saved login.
    if t.get('report_to') == 'wandb' and not os.environ.get('WANDB_API_KEY'):
        import wandb

        api_key = wandb.Api().api_key
        if not api_key:
            raise RuntimeError('W&B credentials missing; run wandb login in the training environment')
        os.environ['WANDB_API_KEY'] = api_key

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    accumulation_steps = t.get('gradient_accumulation_steps', 1)
    # Sequences per optimizer step. EMO splits each rank's share into microbatches.
    global_batch_size = t['micro_batch_size'] * world_size * accumulation_steps
    if cfg.get('pipeline') and global_batch_size != cfg['pipeline']['global_batch_size']:
        raise ValueError('Runtime batching changed the canonical global batch size')
    validation_meta = cfg.get('validation', {}).get('meta', meta['eval'])
    validation_path = Path(validation_meta['output']) if cfg.get('validation') else data_dir / 'eval.bin'
    paths = [data_dir / 'train.bin', validation_path]
    if any(p.stat().st_size // 4 < seq_length for p in paths):
        raise ValueError('Insufficient train/eval tokens for seq_length')
    work_dir = output_dir / 'work'

    prepare_training_environment(seed=t['seed'])
    try:
        torch.set_float32_matmul_precision('highest' if recipe == EMO_RECIPE else 'high')
        tokenizer = TokenizerConfig(vocab_size=meta['vocab_size'], eos_token_id=meta['eos_token_id'],
                                    pad_token_id=meta.get('pad_token_id', meta['eos_token_id']))
        model_cfg = TransformerConfig.llama_like(
            vocab_size=tokenizer.padded_vocab_size(),
            attn_backend=AttentionBackendName.torch, **cfg['model'])
        module_cfg = train_module_config(cfg, world_size, recipe=recipe)
        model = model_cfg.build(init_device='meta')
        module = module_cfg.build(model)
        dataset_kwargs = dict(sequence_length=seq_length, tokenizer=tokenizer,
                              work_dir=str(work_dir), dtype='uint32')
        dataset = training_dataset(data_dir, meta, data_loader=t['data_loader'],
                                   repetition_filter=repetition_filter, **dataset_kwargs)
        first_epoch_steps = len(dataset) // global_batch_size
        extends_epoch_one = (t['max_steps'] > first_epoch_steps if t.get('max_steps') is not None
                             else t.get('epochs', 1) > 1)
        if t['data_loader'] == 'repack' and extends_epoch_one:
            dataset.validate_document_metadata()
        loader = NumpyDataLoaderConfig(
            global_batch_size=global_batch_size * seq_length, seed=t['seed'],
            num_workers=t.get('num_workers', 2)).build(dataset, dp_process_group=module.dp_process_group)
        trainer_cfg = TrainerConfig(
            save_folder=str(output_dir), work_dir=str(work_dir),
            max_duration=Duration.steps(t['max_steps']) if t.get('max_steps') else Duration.epochs(t['epochs']),
            metrics_collect_interval=t.get('logging_steps', 10),
            callbacks={
                'checkpointer': WSDCheckpointerCallback(
                    save_interval=t.get('save_steps', 10000), save_async=True,
                    warmup_steps=t['warmup_steps'], decay_fraction=t['decay_fraction'],
                    decay_horizons=t.get('decay_horizons', [])),
                'config_saver': ConfigSaverCallback(),
                'report_metrics': ReportMetricsCallback(
                    output_dir=str(output_dir),
                    train_tokens_per_byte=meta['train']['tokens'] / meta['train']['bytes'],
                    eval_tokens_per_byte=validation_meta['tokens'] / validation_meta['bytes']),
                'wandb': WandBCallback(enabled=t.get('report_to') == 'wandb',
                                      project=t.get('wandb_project', 'tokenizer-scaling'),
                                      name=cli.run_name or output_dir.name),
                'lm_evaluator': LMEvaluatorCallbackConfig(
                    eval_dataset=NumpyPaddedFSLDatasetConfig(paths=[str(paths[1])], metadata=[{'label': 'validation'}], **dataset_kwargs),
                    eval_interval=t.get('eval_steps', 10000), eval_on_finish=True),
            })
        trainer_cfg.callbacks['training_recipe'] = TrainingRecipeCallback(
            cfg, model_cfg, module_cfg, output_dir, recipe, dataset=dataset)
        if t.get('health_check') and recipe == LEGACY_RECIPE:
            trainer_cfg.callbacks['loss_health'] = LossHealthCallback(str(output_dir), t['warmup_steps'])
        if cfg.get('runtime_batch'):
            trainer_cfg.callbacks['runtime_batch'] = RuntimeBatchEvidenceCallback(
                str(output_dir), cfg['runtime_batch'])
        if t.get('branch_from'):
            from olmo_core.train import LoadStrategy
            parent = Path(t['branch_from'])
            boundary = t['max_steps'] - round(t['max_steps'] * t['decay_fraction'])
            trainer_cfg.load_path = str(parent / f'step{boundary}')
            trainer_cfg.load_strategy = LoadStrategy.always
            trainer_cfg.load_optim_state = trainer_cfg.load_trainer_state = True
        trainer = trainer_cfg.build(module, loader)
        trainer.callbacks['config_saver'].config = cfg
        if cli.resume_from:
            checkpoint = cli.resume_from.resolve()
            if checkpoint.parent != output_dir or not checkpoint.name[4:].isdigit():
                raise ValueError('Resume checkpoint must be a step directory in this run output')
            trainer.load_checkpoint(checkpoint, load_optim_state=True, load_trainer_state=True)
            if trainer.global_step != int(checkpoint.name[4:]) or not module.optim.state:
                raise RuntimeError('Full-state checkpoint restoration failed verification')
            if get_rank() == 0:
                from scripts.pipeline import write_json
                record = {'checkpoint': str(checkpoint), 'step': trainer.global_step,
                          'epoch': trainer.epoch, 'batches_processed': loader.batches_processed,
                          'global_train_tokens_seen': trainer.global_train_tokens_seen,
                          'peak_learning_rate': t['learning_rate'],
                          'restored_optimizer_lr': module.optim.param_groups[0]['lr'],
                          'state': ['model', 'optimizer', 'trainer', 'data_loader', 'rng']}
                write_json(output_dir / 'resume_verified.json', record)
                print('RESUME_VERIFIED ' + json.dumps(record), flush=True)
        if get_rank() == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'tokenizer.json').write_bytes((data_dir / 'tokenizer.json').read_bytes())
            print(f'seq_length={seq_length} sequences_per_step={global_batch_size} '
                  f'tokens_per_step={global_batch_size * seq_length}', flush=True)
        optimizer_groups = verify_optimizer_groups(model, module.optim, t)
        if get_rank() == 0:
            from scripts.pipeline import write_json
            write_json(output_dir / 'optimizer_groups.json', optimizer_groups)
            print('OPTIMIZER_GROUPS ' + json.dumps(optimizer_groups), flush=True)
        started = time.monotonic()
        trainer.fit()
        if trainer.global_step < trainer.max_steps or (output_dir / 'health_stop.json').is_file():
            raise RuntimeError('Training incomplete or unhealthy; do not convert or evaluate as a completed trial')
        if get_rank() == 0:
            from scripts.pipeline import write_json
            write_json(output_dir / 'training_metrics.json', {
                'steps': trainer.global_step, 'max_steps': trainer.max_steps,
                'fit_seconds_this_attempt': time.monotonic() - started,
                'parameters': sum(p.numel() for p in model.parameters()),
                'flops_per_token': model.num_flops_per_token(seq_length),
                'latest_metrics': trainer.callbacks['report_metrics'].latest,
                'sequence_length': seq_length, 'global_batch_size': global_batch_size,
                'data_loader': t['data_loader'],
                'optimizer_groups': optimizer_groups,
                'resolved_training_config': json.loads((output_dir / 'resolved_training_config.json').read_text()),
                **({'runtime_batch': cfg['runtime_batch'],
                    'runtime_batch_verified': json.loads((output_dir / 'runtime_batch_verified.json').read_text())}
                   if cfg.get('runtime_batch') else {}),
                'pre_decay_checkpoint': json.loads((output_dir / 'pre_decay_checkpoint.json').read_text()),
            })
    finally:
        teardown_training_environment()


if __name__ == '__main__':
    main()
