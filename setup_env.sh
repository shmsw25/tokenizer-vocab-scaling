#!/usr/bin/env bash
# Run inside a Python 3.12 environment. CPU jobs install no PyTorch/CUDA packages.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-all}"
case "$mode" in
    cpu) python -m pip install -r scripts/requirements-cpu.txt ;;
    train|all)
        python scripts/verify_emo.py
        python -m pip install -r requirements.txt
        python -m pip install -e './EMO[wandb]'
        if [[ "$mode" == all ]]; then
            python -m pip install -r scripts/requirements-launch.txt
        fi
        ;;
    *) echo "Usage: $0 [cpu|train|all]" >&2; exit 2 ;;
esac
