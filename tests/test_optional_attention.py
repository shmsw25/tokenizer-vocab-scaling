"""A broken optional CUDA extension must not prevent PyTorch model startup."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('failure', [
    "import subprocess; raise subprocess.CalledProcessError(1, \"ldconfig -p | grep 'libcudnn.so'\")",
    "raise OSError('libcudnn.so: cannot open shared object file')",
])
def test_torch_model_with_broken_transformer_engine(tmp_path, failure):
    package = tmp_path / 'transformer_engine'
    package.mkdir()
    (package / '__init__.py').write_text(failure)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(root), str(root / 'EMO/src')]))
    script = '''
import torch
import train
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.te_attn_api import has_te_attn
from olmo_core.nn.transformer import TransformerConfig
assert not has_te_attn()
try:
    AttentionBackendName.te.assert_supported()
except RuntimeError:
    pass
else:
    raise AssertionError('Explicit TE selection must still fail')
model = TransformerConfig.llama_like(d_model=16, n_layers=1, n_heads=2,
    vocab_size=128, tie_word_embeddings=True,
    attn_backend=AttentionBackendName.torch).build(init_device='cpu')
model.init_weights()
output = model(torch.arange(8).reshape(1, 8))
assert torch.isfinite(output).all()
output.float().sum().backward()
assert model.embeddings.weight.grad is not None
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=root, env=env,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Transformer Engine attention unavailable' in result.stderr
