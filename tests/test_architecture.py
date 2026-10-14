import json
from pathlib import Path
import pytest

@pytest.mark.parametrize('name',['small','medium','medium_large'])
def test_study_models_are_emo_prenorm_without_qk_norm(name):
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.block import TransformerBlock
    from olmo_core.nn.attention import AttentionBackendName
    config=json.loads((Path(__file__).resolve().parents[1]/'configs'/f'{name}.json').read_text())
    model=TransformerConfig.llama_like(vocab_size=32000,attn_backend=AttentionBackendName.torch,**config['model']).build(init_device='meta')
    for block in model.blocks.values():
        assert type(block) is TransformerBlock
        assert block.attention.q_norm is None and block.attention.k_norm is None
    assert model.embeddings.weight is model.lm_head.w_out.weight
