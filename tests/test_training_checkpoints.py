"""Exercise the WSD boundary against EMO and a real CPU checkpoint round trip."""
import json
import gzip
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.training_callbacks import ReportMetricsCallback, WSDCheckpointerCallback, pre_decay_step


@pytest.mark.parametrize('total', [1000, 1005, 1015, 10001])
def test_pre_decay_boundary_matches_scheduler(total):
    from olmo_core.optim import WSD
    scheduler = WSD(warmup=96, decay_fraction=0.1, decay_min_lr=0.0)
    boundary = pre_decay_step(total, 96, 0.1)
    assert scheduler.get_lr(1.0, boundary, total) == pytest.approx(1.0)
    assert scheduler.get_lr(1.0, boundary + 1, total) < 1.0
    assert pre_decay_step(10, 96, 0.1) is None
    assert pre_decay_step(4, 1, 0.1) is None


@pytest.mark.parametrize('repack', [False, True])
def test_full_state_pre_decay_checkpoint_and_resume(tmp_path, repack):
    from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.optim import AdamWConfig, WSD
    from olmo_core.train import Duration, TrainerConfig
    from olmo_core.train.callbacks import ConfigSaverCallback
    from olmo_core.train.train_module import TransformerTrainModuleConfig

    tokens = tmp_path / 'tokens.bin'
    # Repack case crosses two epoch boundaries and checkpoints in epoch 2.
    (np.arange(64 if repack else 320, dtype=np.uint32) % 128).tofile(tokens)
    documents = tmp_path / 'documents.csv.gz'
    if repack:
        with gzip.open(documents, 'wt') as source:
            source.writelines(f'{start},{start + 16}\n' for start in range(0, 64, 16))
    output = tmp_path / 'model'

    def build():
        model = TransformerConfig.llama_like(
            d_model=16, n_layers=1, n_heads=2, vocab_size=128,
            tie_word_embeddings=True, attn_backend=AttentionBackendName.torch).build(init_device='cpu')
        module = TransformerTrainModuleConfig(
            rank_microbatch_size=16, max_sequence_length=8,
            optim=AdamWConfig(lr=0.001, weight_decay=0.1, fused=False),
            scheduler=WSD(warmup=2, decay_fraction=0.3, decay_min_lr=0),
            compile_model=False).build(model, device=torch.device('cpu'))
        dataset = NumpyFSLDatasetConfig(
            paths=[str(tokens)], sequence_length=8, dtype='uint32', work_dir=str(tmp_path / 'data'),
            dynamic_repacking=repack, num_documents=4,
            document_metadata_path=str(documents),
            tokenizer=TokenizerConfig(vocab_size=128, eos_token_id=127, pad_token_id=127)).build()
        loader = NumpyDataLoaderConfig(global_batch_size=16, seed=42, num_workers=0).build(dataset)
        callback = WSDCheckpointerCallback(
            warmup_steps=2, decay_fraction=0.3, save_interval=7,
            pre_train_checkpoint=False, save_async=True)
        trainer = TrainerConfig(
            save_folder=str(output), work_dir=str(tmp_path / 'work'), device='cpu',
            max_duration=Duration.steps(10), metrics_collect_interval=1,
            callbacks={'checkpointer': callback, 'config_saver': ConfigSaverCallback(),
                       'report_metrics': ReportMetricsCallback(str(output), 0.25, 0.24)},
        ).build(module, loader)
        trainer.callbacks['config_saver'].config = {'test': True}
        def unexpected_async_save(**kwargs):
            raise AssertionError('Boundary/final checkpoints must complete synchronously')
        trainer.save_checkpoint_async = unexpected_async_save
        return trainer, callback

    trainer, callback = build()
    trainer.fit()
    metrics = json.loads((output / 'report_metrics.json').read_text())
    assert metrics['train_step'] == 10
    assert metrics['train_bpb'] == pytest.approx(metrics['train_ce'] * 0.25 / np.log(2))
    record_path = output / 'pre_decay_checkpoint.json'
    record = json.loads(record_path.read_text())
    assert record['status'] == 'saved' and record['step'] == 7
    assert record['first_reduced_lr_step'] == 8
    checkpoint = Path(record['path'])
    state = torch.load(checkpoint / 'train/rank0.pt', weights_only=False)
    assert state['global_step'] == 7
    assert state['data_loader']['batches_processed'] == (3 if repack else 7)
    if repack:
        assert state['data_loader']['epoch'] == 2
        assert state['data_loader']['repacking']['document_metadata_digest']
    assert state['rng']
    assert json.loads((checkpoint / '.metadata.json').read_text())['ephemeral'] is False
    assert (output / 'step10/model_and_optim/.metadata').is_file()
    assert callback._checkpoints.count(str(checkpoint)) == 1  # interval also equals 7

    # Restore actual model/optimizer/dataloader state at the branch point.
    resumed, _ = build()
    resumed.load_checkpoint(checkpoint)
    assert resumed.global_step == 7
    assert resumed.data_loader.batches_processed == (3 if repack else 7)
    assert resumed.train_module.optim.param_groups[0]['lr'] == pytest.approx(0.001)
    assert resumed.train_module.optim.state
    assert resumed.train_module.scheduler.get_lr(0.001, 8, 10) < 0.001
    # Restarting a completed run reconstructs a missing index without deleting the checkpoint.
    record_path.unlink()
    finished, _ = build()
    finished.fit()
    assert json.loads(record_path.read_text()) == record


def test_intermediate_decay_branch_restores_full_state_and_leaves_trunk_intact(tmp_path):
    from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.optim import AdamWConfig, WSD
    from olmo_core.train import Duration, LoadStrategy, TrainerConfig
    from olmo_core.train.callbacks import ConfigSaverCallback
    from olmo_core.train.train_module import TransformerTrainModuleConfig

    tokens = tmp_path / 'tokens.npy'
    np.arange(1024, dtype=np.uint32).__mod__(127).tofile(tokens)
    def build(output, steps, source=None, horizons=()):
        torch.manual_seed(42)
        model = TransformerConfig.llama_like(d_model=16,n_layers=1,n_heads=2,vocab_size=128,
            tie_word_embeddings=True,attn_backend=AttentionBackendName.torch).build(init_device='cpu')
        module = TransformerTrainModuleConfig(rank_microbatch_size=16,max_sequence_length=8,
            optim=AdamWConfig(lr=.001,weight_decay=.1,fused=False),
            scheduler=WSD(warmup=2,decay_fraction=.1,decay_min_lr=0),compile_model=False).build(model,device=torch.device('cpu'))
        dataset = NumpyFSLDatasetConfig(paths=[str(tokens)],sequence_length=8,dtype='uint32',
            work_dir=str(tmp_path/'data'),tokenizer=TokenizerConfig(vocab_size=128,eos_token_id=127,pad_token_id=127)).build()
        loader = NumpyDataLoaderConfig(global_batch_size=16,seed=42,num_workers=0).build(dataset)
        callback = WSDCheckpointerCallback(warmup_steps=2,decay_fraction=.1,save_interval=1000,
            pre_train_checkpoint=False,save_async=False,decay_horizons=list(horizons))
        config = TrainerConfig(save_folder=str(output),work_dir=str(tmp_path/'work'),device='cpu',
            max_duration=Duration.steps(steps),metrics_collect_interval=1,
            load_path=str(source) if source else None,load_strategy=LoadStrategy.always if source else LoadStrategy.if_available,
            load_optim_state=True,load_trainer_state=True,
            callbacks={'checkpointer':callback,'config_saver':ConfigSaverCallback()})
        trainer=config.build(module,loader);trainer.callbacks['config_saver'].config={'test':True}
        return trainer
    trunk = build(tmp_path/'trunk',20,horizons=[.5]);trunk.fit()
    record=json.loads((tmp_path/'trunk/decay_branches.json').read_text())['0.5']
    assert record['step']==9 and record['total_steps']==10
    checkpoint=Path(record['path'])
    assert (checkpoint/'model_and_optim/.metadata').is_file()
    state=torch.load(checkpoint/'train/rank0.pt',weights_only=False)
    assert state['global_step']==9 and state['data_loader']['batches_processed']==9 and state['rng']
    parent_bytes=(checkpoint/'train/rank0.pt').read_bytes()
    branch=build(tmp_path/'branch',10,source=checkpoint);branch.fit()
    assert branch.global_step==10 and branch.data_loader.batches_processed==10
    assert branch.train_module.optim.state
    assert branch.train_module.optim.param_groups[0]['lr']==pytest.approx(0.)
    saved=json.loads((tmp_path/'branch/pre_decay_checkpoint.json').read_text())
    assert saved['step']==9 and Path(saved['path']).parent==tmp_path/'branch'
    assert (checkpoint/'train/rank0.pt').read_bytes()==parent_bytes
    assert trunk.global_step==20
