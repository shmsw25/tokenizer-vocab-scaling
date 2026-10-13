"""Checkpoint policy for retaining the full-LR branch point of a WSD run."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from olmo_core.distributed.utils import get_rank
from olmo_core.train.callbacks import Callback, CheckpointerCallback


@dataclass
class ReportMetricsCallback(Callback):
    """Publish loss observations for the report without changing training."""
    output_dir: str
    train_tokens_per_byte: float
    eval_tokens_per_byte: float
    latest: dict = field(default_factory=dict)

    def pre_train(self):
        path = Path(self.output_dir) / 'report_metrics.json'
        if path.is_file():
            self.latest = json.loads(path.read_text())

    def log_metrics(self, step, metrics):
        if get_rank() != 0:
            return
        changed = False
        for name, value in metrics.items():
            if name == 'train/CE loss':
                label, ratio = 'train', self.train_tokens_per_byte
            elif name.startswith('eval/') and name.endswith('/validation/CE loss'):
                label, ratio = 'validation', self.eval_tokens_per_byte
            else:
                continue
            if not math.isfinite(float(value)) or step < self.latest.get(f'{label}_step', 0):
                continue
            self.latest.update({f'{label}_ce': float(value),
                                f'{label}_bpb': float(value) * ratio / math.log(2),
                                f'{label}_step': step})
            changed = True
        if changed:
            from scripts.pipeline import write_json
            self.latest['step'] = max(step, self.latest.get('step', 0))
            write_json(Path(self.output_dir) / 'report_metrics.json', self.latest)
            print('REPORT_METRICS ' + json.dumps(self.latest, sort_keys=True), flush=True)


def pre_decay_step(total_steps, warmup_steps, decay_fraction):
    # Match WSD.get_lr() exactly, including Python's rounding at half steps.
    decay_steps = round(total_steps * decay_fraction)
    boundary = total_steps - decay_steps
    if decay_steps == 0 or boundary < max(1, warmup_steps):
        return None  # A short smoke run may never have a full-LR/decay transition.
    return boundary


@dataclass
class WSDCheckpointerCallback(CheckpointerCallback):
    warmup_steps: int = 96
    decay_fraction: float = 0.1
    _boundary = None
    _total_steps = None

    def post_checkpoint_loaded(self, path):
        # EMO otherwise tries to re-save a completed run's final checkpoint.
        # Only count a checkpoint already in this run's output directory.
        expected = Path(self.save_folder) / self.checkpointer.checkpoint_dirname(self.step)
        if Path(path) == expected:
            self._latest_checkpoint_step = self.step
            self._latest_checkpoint_path = str(path)

    def pre_train(self):
        if not self.enabled:
            raise ValueError('WSD training requires checkpointing to be enabled')
        self._total_steps = self.trainer.max_steps
        if self._total_steps is None:
            raise ValueError('WSD checkpointing requires a known training horizon')
        self._boundary = pre_decay_step(self._total_steps, self.warmup_steps, self.decay_fraction)
        if self._boundary is not None:
            self.fixed_steps = sorted(set(self.fixed_steps or []) | {self._boundary})
        super().pre_train()
        if self._boundary is None:
            self._write_record({'status': 'not_applicable',
                                'reason': 'No full-LR step before decay in this short run'})
        elif self.step >= self._boundary:
            # Also recover the index after a restart between the save and index write.
            self._record_saved_checkpoint()
        if get_rank() == 0:
            print(f'WSD pre-decay checkpoint step: {self._boundary}; '
                  f'total steps: {self._total_steps}', flush=True)

    def _save_checkpoint(self, save_async=None, ephemeral=False):
        at_boundary = self._boundary is not None and self.step == self._boundary
        path = super()._save_checkpoint(
            save_async=False if at_boundary else save_async,
            ephemeral=False if at_boundary else ephemeral)
        if at_boundary:
            # Finish the full-state write before the first reduced-LR optimizer step.
            self._record_saved_checkpoint()
        return path

    def _write_record(self, record):
        if get_rank() == 0:
            from scripts.pipeline import write_json
            write_json(Path(self.save_folder) / 'pre_decay_checkpoint.json', {
                'scheduler': 'WSD', 'total_steps': self._total_steps,
                'warmup_steps': self.warmup_steps, 'decay_fraction': self.decay_fraction,
                **record,
            })

    def _record_saved_checkpoint(self):
        path = Path(self.save_folder) / self.checkpointer.checkpoint_dirname(self._boundary)
        if not self.checkpointer.dir_is_checkpoint(path):
            raise RuntimeError(f'Required full-state pre-decay checkpoint is missing: {path}')
        self._write_record({'status': 'saved', 'path': str(path), 'step': self._boundary,
                            'first_reduced_lr_step': self._boundary + 1,
                            'state': ['model', 'optimizer', 'trainer', 'data_loader', 'rng']})
