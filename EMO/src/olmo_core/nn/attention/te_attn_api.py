import logging
import subprocess


try:
    import transformer_engine.pytorch as te  # type: ignore
except ImportError:
    te = None
except (OSError, subprocess.CalledProcessError) as error:
    # TE is optional. Installed packages with missing CUDA libraries must not
    # prevent explicitly selected PyTorch/FlashAttention backends from loading.
    logging.getLogger(__name__).warning("Transformer Engine attention unavailable: %s", error)
    te = None


def has_te_attn() -> bool:
    """Check if Transformer Engine attention is available."""
    return te is not None


TEDotProductAttention = te.DotProductAttention if te is not None else None
