#!/usr/bin/env bash
# The launcher uploads a committed snapshot, including EMO, as a Beaker dataset.
set -euo pipefail
mkdir -p /results
task_dir="$(mktemp -d /tmp/tokenizer-project.XXXXXX)"
tar -xzf /code/repo.tar.gz -C "$task_dir"
cd "$task_dir"
python3 -m venv --system-site-packages "$task_dir/.venv"
source "$task_dir/.venv/bin/activate"
mode=cpu
case "$1" in train-model|convert-model|evaluate-model|validate-model) mode=train ;; esac
bash setup_env.sh "$mode"
if [[ "${TOKENIZER_CHAIN_EVALUATION:-0}" == 1 ]]; then
    exec bash scripts/run_model.sh "$@"
fi
exec python scripts/pipeline.py "$@"
