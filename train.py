"""Train a dense Llama-like language model on data tokenized by tokenize_data.py.

    torchrun --standalone --nproc_per_node=2 train.py \
        --config configs/small.json --data-dir tokenized/32k --output-dir runs/small_32k

The model and optimizer settings come from the config JSON. The vocabulary
size, EOS token and tokenizer come from the tokenized data directory.

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
import os
import time
from pathlib import Path

import torch

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, NumpyPaddedFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.utils import get_rank
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, WSD
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import ConfigSaverCallback, LMEvaluatorCallbackConfig, WandBCallback
from olmo_core.train.train_module import TransformerDataParallelConfig, TransformerTrainModuleConfig
from scripts.training_callbacks import WSDCheckpointerCallback


def load_meta(data_dir):
    meta_path = Path(data_dir) / 'meta.json'
    if not meta_path.is_file():
        raise FileNotFoundError(f'{meta_path} is missing; run tokenize_data.py first')
    return json.loads(meta_path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True, help='Output directory of tokenize_data.py')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seq-length', type=int)
    parser.add_argument('--match-context-of', type=Path, metavar='OTHER_DIR',
                        help='Tokenized directory of the reference tokenizer')
    parser.add_argument('--micro-batch-size', type=int, help='Sequences per GPU per forward pass')
    parser.add_argument('--gradient-accumulation-steps', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--max-steps', type=int, help='Stop after this many steps instead of the configured epochs')
    parser.add_argument('--run-name', help='W&B run name; defaults to the output directory name')
    cli = parser.parse_args()

    cfg = json.loads(cli.config.read_text())
    t = cfg['training']
    for key in ('micro_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'max_steps'):
        if getattr(cli, key) is not None:
            t[key] = getattr(cli, key)
    data_dir, output_dir = cli.data_dir.resolve(), cli.output_dir.resolve()
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
    paths = [data_dir / 'train.bin', data_dir / 'eval.bin']
    if any(p.stat().st_size // 4 < seq_length for p in paths):
        raise ValueError('Insufficient train/eval tokens for seq_length')
    work_dir = output_dir / 'work'

    prepare_training_environment(seed=t['seed'])
    try:
        torch.set_float32_matmul_precision('high')
        tokenizer = TokenizerConfig(vocab_size=meta['vocab_size'], eos_token_id=meta['eos_token_id'],
                                    pad_token_id=meta['eos_token_id'])
        model_cfg = TransformerConfig.llama_like(
            vocab_size=tokenizer.padded_vocab_size(),
            attn_backend=AttentionBackendName.torch, **cfg['model'])
        module_cfg = TransformerTrainModuleConfig(
            dp_config=TransformerDataParallelConfig(name=DataParallelType.ddp) if world_size > 1 else None,
            rank_microbatch_size=t['micro_batch_size'] * seq_length,
            max_sequence_length=seq_length,
            optim=AdamWConfig(lr=t['learning_rate'], weight_decay=t['weight_decay'], fused=True),
            scheduler=WSD(warmup=t['warmup_steps'],
                          decay_fraction=t['decay_fraction'], decay_min_lr=t['decay_min_lr']),
            max_grad_norm=1.0, compile_model=t['compile'],
            autocast_precision=DType.bfloat16)
        model = model_cfg.build(init_device='meta')
        module = module_cfg.build(model)
        dataset_kwargs = dict(sequence_length=seq_length, tokenizer=tokenizer,
                              work_dir=str(work_dir), dtype='uint32')
        dataset = NumpyFSLDatasetConfig(paths=[str(paths[0])], **dataset_kwargs).build()
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
                    warmup_steps=t['warmup_steps'], decay_fraction=t['decay_fraction']),
                'config_saver': ConfigSaverCallback(),
                'wandb': WandBCallback(enabled=t.get('report_to') == 'wandb',
                                      project=t.get('wandb_project', 'tokenizer-scaling'),
                                      name=cli.run_name or output_dir.name),
                'lm_evaluator': LMEvaluatorCallbackConfig(
                    eval_dataset=NumpyPaddedFSLDatasetConfig(paths=[str(paths[1])], metadata=[{'label': 'validation'}], **dataset_kwargs),
                    eval_interval=t.get('eval_steps', 10000), eval_on_finish=True),
            })
        trainer = trainer_cfg.build(module, loader)
        trainer.callbacks['config_saver'].config = cfg
        if get_rank() == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'tokenizer.json').write_bytes((data_dir / 'tokenizer.json').read_bytes())
            print(f'seq_length={seq_length} sequences_per_step={global_batch_size} '
                  f'tokens_per_step={global_batch_size * seq_length}', flush=True)
        started = time.monotonic()
        trainer.fit()
        if get_rank() == 0:
            from scripts.pipeline import write_json
            write_json(output_dir / 'training_metrics.json', {
                'steps': trainer.global_step, 'max_steps': trainer.max_steps,
                'fit_seconds_this_attempt': time.monotonic() - started,
                'parameters': sum(p.numel() for p in model.parameters()),
                'sequence_length': seq_length, 'global_batch_size': global_batch_size,
                'pre_decay_checkpoint': json.loads((output_dir / 'pre_decay_checkpoint.json').read_text()),
            })
    finally:
        teardown_training_environment()


if __name__ == '__main__':
    main()
