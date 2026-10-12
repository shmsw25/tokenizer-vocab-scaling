#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-$repo/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin=python3
exec "$python_bin" "$repo/scripts/submit.py" tokenize-data "$@"
