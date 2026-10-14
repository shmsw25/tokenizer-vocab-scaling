#!/usr/bin/env bash
# One Beaker allocation: train -> CPU checkpoint conversion -> single-GPU eval.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
first_stage="$1"
shift
case "$first_stage" in
    train-model|convert-model) ;;
    *) echo "Usage: $0 {train-model|convert-model} MODEL_ARGUMENTS..." >&2; exit 2 ;;
esac
reporter_pid=""
cleanup() {
    if [[ -n "$reporter_pid" ]]; then
        kill "$reporter_pid" 2>/dev/null || true
        wait "$reporter_pid" 2>/dev/null || true
        reporter_pid=""
    fi
}
trap cleanup EXIT
run_stage() {
    local stage="$1"
    shift
    python scripts/chain_status.py "$stage"
    python scripts/chain_status.py "$stage" --watch &
    reporter_pid=$!
    # A failed stage exits the shell; later stages must not run.
    python scripts/pipeline.py "$stage" "$@"
    cleanup
    python scripts/chain_status.py "$stage"
}
if [[ "$first_stage" == train-model ]]; then
    run_stage train-model "$@"
fi
CUDA_VISIBLE_DEVICES="" run_stage convert-model "$@"
# Respect Beaker's assigned device mapping, exposing only its first device.
eval_device="${CUDA_VISIBLE_DEVICES-0}"
eval_device="${eval_device%%,*}"
CUDA_VISIBLE_DEVICES="$eval_device" run_stage evaluate-model "$@"
