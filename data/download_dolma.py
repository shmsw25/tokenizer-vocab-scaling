"""Run: python -m pip install -r requirements.txt && python download_dolma.py"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

import fsspec

from datasets import load_dataset
from huggingface_hub import HfApi

if "zstd" not in fsspec.available_compressions():
    raise SystemExit(
        "Dolma requires zstandard. Install it with: python -m pip install zstandard"
    )

parser = argparse.ArgumentParser(description="Download the pinned proportional 250GB Dolma sample")
parser.add_argument("--output", type=Path, default=Path("dolma3_250gb.jsonl"))
args = parser.parse_args()
estimates_dir = Path(__file__).resolve().parent

repo = "allenai/dolma3_mix-6T"
revision = json.loads((estimates_dir / "dolma_text_estimate_metadata.json").read_text())["revision"]
with (estimates_dir / "dolma_topic_text_estimates.csv").open() as f:
    topics = list(csv.DictReader(f))
estimated_total = sum(float(row["estimated_text_bytes"]) for row in topics)
files_by_topic = {row["source_topic"]: [] for row in topics}
for path in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision):
    if path.startswith("data/") and path.endswith(".zst"):
        topic = re.sub(r"-part\d+$", "", re.sub(r"-\d+$", "", path.split("/")[1]))
        files_by_topic[topic].append(path)
rng = random.Random(42)

target_bytes = 250_000_000_000  # UTF-8 document text, excluding metadata.
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
partial = output.with_suffix(".jsonl.part")
text_bytes = 0
count = 0
budgets = [round(target_bytes * float(row["estimated_text_bytes"]) / estimated_total) for row in topics]
saved_bytes = [0] * len(topics)
saved_counts = [0] * len(topics)
if output.exists():
    raise SystemExit(f"{output} already exists")
if partial.exists():
    print("Scanning existing partial file to recover progress...", flush=True)
    topic_index = 0
    with partial.open("r", encoding="utf-8") as saved:
        for line in saved:
            if not line.endswith("\n"):
                raise SystemExit("Partial file has an incomplete final line; refusing to append")
            size = len(json.loads(line)["text"].encode("utf-8"))
            saved_bytes[topic_index] += size
            saved_counts[topic_index] += 1
            text_bytes += size
            count += 1
            if saved_bytes[topic_index] >= budgets[topic_index]:
                topic_index += 1
            if count % 1_000_000 == 0:
                print(f"Recovered {count:,} records, {text_bytes / 1e9:.3f} GB", flush=True)
    print(f"Resuming at {count:,} records, {text_bytes / 1e9:.3f} GB", flush=True)

with partial.open("a", encoding="utf-8") as f:
    for topic_index, row in enumerate(topics):
        topic = row["source_topic"]
        budget = budgets[topic_index]
        topic_bytes = saved_bytes[topic_index]
        skip = saved_counts[topic_index]
        files = sorted(files_by_topic[topic])
        rng.shuffle(files)
        if topic_bytes >= budget:
            continue
        print(f"{topic}: target {budget / 1e9:.3f} GB", flush=True)
        for path in files:
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
            dataset = load_dataset("json", data_files=url, split="train", streaming=True)
            for entry in dataset:
                if skip:
                    skip -= 1
                    continue
                # Hugging Face may parse JSON timestamp strings into datetime objects.
                f.write(json.dumps(entry, ensure_ascii=False, default=lambda value: value.isoformat()) + "\n")
                size = len(entry["text"].encode("utf-8"))
                text_bytes += size
                topic_bytes += size
                count += 1
                if count % 1_000 == 0:
                    print(f"Downloaded {count:,} records, {text_bytes / 1e9:.3f} / 250 GB of text", flush=True)
                if topic_bytes >= budget:
                    break
            if topic_bytes >= budget:
                break
        if topic_bytes < budget:
            raise RuntimeError(f"{topic}: exhausted shards before reaching its budget")

partial.replace(output)
print(f"Saved {count:,} records ({text_bytes:,} text bytes) to {output}")
