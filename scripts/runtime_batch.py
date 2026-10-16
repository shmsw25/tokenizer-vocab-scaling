"""Choose execution microbatches from assigned GPUs before starting torchrun.

The launch configuration and output fingerprint remain unchanged. This plan
only partitions each rank's fixed global-batch share into smaller microbatches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

GPU_CLUSTERS = ('ai2/jupiter', 'ai2/saturn', 'ai2/neptune')
GPU_TYPES = {'GPU_TYPE_NVIDIA_H100', 'GPU_TYPE_NVIDIA_A100_80GB', 'GPU_TYPE_NVIDIA_L40S'}


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def device_family(device):
    name = device['name'].upper()
    family = next((kind for kind in ('H100', 'A100', 'L40S') if kind in name), None)
    minimum_gib = {'H100': 75, 'A100': 75, 'L40S': 44}
    if family is None or device['total_memory'] < minimum_gib[family] * 1024**3:
        raise ValueError(f'Unsupported GPU or insufficient capacity: {device}')
    return family


def make_plan(config, model_size, vocab_size, devices):
    world_size = config['pipeline']['gpus']
    batch = config['pipeline']['global_batch_size']
    canonical_micro = config['training']['micro_batch_size']
    if world_size != 8 or len(devices) != world_size:
        raise ValueError('Training requires exactly eight visible GPUs on one node')
    if batch != canonical_micro * world_size * config['training']['gradient_accumulation_steps']:
        raise ValueError('Canonical global batch and accumulation disagree')
    dimensions = {'small': 512, 'medium': 1024, 'medium_large': 2048}
    if model_size not in dimensions or config['model']['d_model'] != dimensions[model_size]:
        raise ValueError('No runtime microbatch policy for this model configuration')
    families = {device_family(device) for device in devices}
    if len(families) != 1:
        raise ValueError('Mixed GPU types within one training allocation are unsupported')
    family = families.pop()
    # Retain canonical microbatches on 80GB devices and start conservatively on
    # 48GB L40S. A changed recipe (including FSDP) needs new runtime memory evidence;
    # historical DDP memory measurements do not verify its fit.
    cap = canonical_micro
    if family == 'L40S':
        cap = {'small': 8, 'medium': 4, 'medium_large': 1 if vocab_size >= 128000 else 2}[model_size]
    candidates = [micro for micro in range(1, min(cap, canonical_micro) + 1)
                  if batch % (world_size * micro) == 0]
    if not candidates:
        raise ValueError('Cannot preserve the global batch with an integral accumulation')
    micro = max(candidates)
    return {'policy_version': 1, 'config_sha256': config_digest(config),
            'model_size': model_size, 'vocab_size': vocab_size, 'gpu_family': family,
            'devices': devices, 'world_size': world_size, 'global_batch_size': batch,
            'canonical_micro_batch_size': canonical_micro, 'micro_batch_size': micro,
            'gradient_accumulation_steps': batch // (world_size * micro)}


def apply_plan(config, plan, world_size, local_rank, device):
    """Verify the immutable recipe and this rank's actual GPU before applying."""
    expected = make_plan(config, plan['model_size'], plan['vocab_size'], plan['devices'])
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Runtime batch plan does not match the launch configuration or policy')
    if world_size != plan['world_size'] or not 0 <= local_rank < world_size:
        raise ValueError('Runtime world size differs from the pre-launch plan')
    if device != plan['devices'][local_rank]:
        raise ValueError('Assigned GPU differs from the pre-launch plan')
    config['training'].update(micro_batch_size=plan['micro_batch_size'],
                              gradient_accumulation_steps=plan['gradient_accumulation_steps'])
    config['runtime_batch'] = plan


def cuda_device(torch, index):
    device = torch.cuda.get_device_properties(index)
    return {'name': device.name, 'total_memory': device.total_memory,
            'compute_capability': [device.major, device.minor]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-size', required=True)
    parser.add_argument('--vocab-size', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from pipeline import write_json
    devices = [cuda_device(torch, index) for index in range(torch.cuda.device_count())]
    plan = make_plan(json.loads(args.config.read_text()), args.model_size, args.vocab_size, devices)
    plan.update(created_at_ns=time.time_ns(), hostname=os.environ.get('BEAKER_NODE_HOSTNAME'),
                source_revision=os.environ.get('TOKENIZER_CODE_REVISION', 'local'))
    write_json(args.output, plan)
    write_json(args.output.parent / 'runtime_batch_history' / f'{plan["created_at_ns"]}.json', plan)
    if Path('/results').is_dir():
        write_json(Path('/results/runtime_batch_plan.json'), plan)
    print('RUNTIME_BATCH_PLAN ' + json.dumps(plan, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
