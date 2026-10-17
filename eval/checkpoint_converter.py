"""Convert trainer_emo.py checkpoints to Hugging Face Llama or Olmo2 models.

Run with the tokenizer-scaling Python environment (torch, transformers, olmo_core).
Pass one or more step directories. Outputs are model_checkpoints/<run>/<step>.
Without arguments, convert the latest saved step of each current 250GB run.
Weights retain float32 precision; optimizer state is not exported. CPU only.
"""
import argparse
import json
import tempfile
from pathlib import Path


def convert(checkpoint, output_root):
    import torch
    from olmo_core.data import TokenizerConfig
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.transformer import TransformerConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaConfig,
                              LlamaForCausalLM, Olmo2Config, Olmo2ForCausalLM,
                              PreTrainedTokenizerFast)

    checkpoint = checkpoint.resolve()
    destination = output_root / checkpoint.parent.name / checkpoint.name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    cfg = json.loads((checkpoint / 'config.json').read_text())
    # Limit conversion to the architecture used by these training runs.
    supported = {'d_model', 'n_layers', 'n_heads', 'hidden_size_multiplier',
                 'block_name', 'qk_norm', 'rope_theta', 'layer_norm_eps',
                 'n_kv_heads', 'tie_word_embeddings', 'init_std', 'init_tied_output'}
    model_options = cfg['model']
    olmo2 = (model_options.get('block_name') == 'reordered_norm'
             and model_options.get('qk_norm') is True)
    llama = (model_options.get('block_name', 'default') == 'default'
             and not model_options.get('qk_norm', False))
    if set(model_options) - supported or not (olmo2 or llama):
        raise ValueError('Unsupported llama_like architecture')
    tied = model_options.get('tie_word_embeddings', False)
    tokenizer_path = checkpoint.parent / 'tokenizer.json'
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing saved tokenizer: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path), eos_token=cfg['eos_token'],
        pad_token=cfg.get('pad_token', cfg['eos_token']), model_max_length=cfg['seq_length'])
    vocab = tokenizer.get_vocab()
    if cfg['eos_token'] not in vocab:
        raise ValueError('EOS token is missing from the saved vocabulary')
    tok_cfg = TokenizerConfig(vocab_size=max(vocab.values()) + 1,
                              eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id)
    model_cfg = TransformerConfig.llama_like(
        vocab_size=tok_cfg.padded_vocab_size(), attn_backend=AttentionBackendName.torch,
        **cfg['model'])
    source = model_cfg.build(init_device='meta')
    source.to_empty(device='cpu')
    load_model_and_optim_state(checkpoint / 'model_and_optim', source, thread_count=4)
    if tied:
        if not torch.equal(source.embeddings.weight, source.lm_head.w_out.weight):
            raise ValueError('Tied checkpoint has unequal input and output embeddings')
        source.lm_head.w_out.weight = source.embeddings.weight
    source.eval()
    hf_common = dict(
        vocab_size=model_cfg.vocab_size, hidden_size=model_cfg.d_model,
        intermediate_size=model_cfg.block.feed_forward.hidden_size,
        num_hidden_layers=model_cfg.n_layers,
        num_attention_heads=cfg['model']['n_heads'],
        num_key_value_heads=cfg['model'].get('n_kv_heads', cfg['model']['n_heads']),
        max_position_embeddings=cfg['seq_length'],
        rms_norm_eps=cfg['model'].get('layer_norm_eps', 1e-5),
        hidden_act='silu', attention_bias=False, mlp_bias=False,
        tie_word_embeddings=tied, bos_token_id=None,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    if olmo2:
        hf_cfg = Olmo2Config(**hf_common, rope_parameters={
            'rope_type': 'default', 'rope_theta': cfg['model'].get('rope_theta', 500000)})
        hf_class = Olmo2ForCausalLM
    else:
        hf_cfg = LlamaConfig(**hf_common, rope_theta=cfg['model'].get('rope_theta', 500000))
        hf_class = LlamaForCausalLM
    mapping = {'embeddings.weight': 'model.embed_tokens.weight',
               'lm_head.norm.weight': 'model.norm.weight',
               'lm_head.w_out.weight': 'lm_head.weight'}
    layer_mapping = {
        'attention.w_q': 'self_attn.q_proj', 'attention.w_k': 'self_attn.k_proj',
        'attention.w_v': 'self_attn.v_proj', 'attention.w_out': 'self_attn.o_proj',
        'feed_forward.w1': 'mlp.gate_proj', 'feed_forward.w2': 'mlp.down_proj',
        'feed_forward.w3': 'mlp.up_proj', 'attention_norm': 'input_layernorm',
        'feed_forward_norm': 'post_attention_layernorm'}
    if olmo2:
        layer_mapping.update({'attention.q_norm': 'self_attn.q_norm',
                              'attention.k_norm': 'self_attn.k_norm',
                              'attention_norm': 'post_attention_layernorm',
                              'feed_forward_norm': 'post_feedforward_layernorm'})
    for i in range(model_cfg.n_layers):
        mapping.update({f'blocks.{i}.{old}.weight': f'model.layers.{i}.{new}.weight'
                        for old, new in layer_mapping.items()})
    state = source.state_dict()
    if set(state) != set(mapping):
        raise ValueError(f'Unexpected model weights: {set(state) ^ set(mapping)}')
    with torch.device('meta'):
        hf = hf_class(hf_cfg)
    hf.load_state_dict({mapping[k]: v for k, v in state.items()}, strict=True, assign=True)
    if tied:
        hf.tie_weights()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after saving, reloading, and checking numerical equivalence.
    with tempfile.TemporaryDirectory(prefix='.converting-', dir=destination.parent) as tmp:
        hf.save_pretrained(tmp, safe_serialization=True)
        tokenizer.save_pretrained(tmp)
        del hf
        loaded = AutoModelForCausalLM.from_pretrained(tmp, local_files_only=True,
                                                    attn_implementation='sdpa').eval()
        loaded_tokenizer = AutoTokenizer.from_pretrained(tmp, local_files_only=True)
        generator = torch.Generator().manual_seed(42)
        cases = [loaded_tokenizer('The quick brown fox jumps over the lazy dog.',
                                  return_tensors='pt').input_ids,
                 torch.randint(tok_cfg.vocab_size, (1, min(64, cfg['seq_length'])), generator=generator),
                 torch.randint(tok_cfg.vocab_size, (1, cfg['seq_length']), generator=generator)]
        max_error = 0.0
        with torch.inference_mode():
            for ids in cases:
                expected = source(ids)
                actual = loaded(ids).logits
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
                max_error = max(max_error, (actual - expected).abs().max().item())
        (Path(tmp) / 'conversion.json').write_text(json.dumps({
            'source_checkpoint': str(checkpoint), 'max_logit_error': max_error,
            'validation_lengths': [ids.shape[1] for ids in cases]}, indent=2) + '\n')
        Path(tmp).rename(destination)
    print(f'Saved {destination} (max logit error {max_error:.3g})', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='*', type=Path, help='Saved step directories')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'model_checkpoints')
    args = parser.parse_args()
    if not args.checkpoints:
        training = Path(__file__).resolve().parent.parent / 'train' / 'model' / 'output'
        for run in sorted(training.glob('outputs_emo_*_dolma3_250gb_2gpu')):
            steps = [p for p in run.glob('step*') if p.name[4:].isdigit()
                     and (p / '.metadata.json').is_file()
                     and (p / 'config.json').is_file()
                     and (p / 'model_and_optim' / '.metadata').is_file()]
            if steps:
                args.checkpoints.append(max(steps, key=lambda p: int(p.name[4:])))
    if not args.checkpoints:
        parser.error('No completed checkpoints found; pass a saved step directory explicitly')
    import torch
    torch.set_num_threads(4)
    for checkpoint in args.checkpoints:
        print(f'Converting {checkpoint}', flush=True)
        convert(checkpoint, args.output_dir.resolve())


if __name__ == '__main__':
    main()
