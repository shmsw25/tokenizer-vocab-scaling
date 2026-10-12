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
[[ "$1" != train-model ]] || mode=train
bash setup_env.sh "$mode"
exec python scripts/pipeline.py "$@"
