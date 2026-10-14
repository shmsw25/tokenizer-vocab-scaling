"""Verify that the vendored EMO files still contain the pinned upstream patch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    provenance = json.loads((ROOT / 'EMO_VENDOR.json').read_text())
    for name, expected in provenance['patched_file_sha256'].items():
        path = ROOT / 'EMO' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Patched EMO file missing or changed: {path}')
    print(f"Verified local EMO {provenance['commit']} with recorded local patches")


if __name__ == '__main__':
    main()
