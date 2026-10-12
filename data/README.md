# Dolma data

Downloads approximately **250 GB of UTF-8 document text** from
`allenai/dolma3_mix-6T`. JSON and metadata make the output file larger than 250 GB.

## Selection

- The release is divided into 62 source/topic groups.
- Text sizes were estimated from 185 randomly selected complete shards: up to
  three per group, sampled without replacement with seed `42`.
- Each group's estimated text size is its total compressed size multiplied by
  the sampled text-to-compressed-byte ratio. The results are in
  `dolma_topic_text_estimates.csv`; the pinned dataset revision and methodology
  are in `dolma_text_estimate_metadata.json`.
- The downloader divides 250 GB proportionally among these estimates. Within
  each group, it shuffles shard order with seed `42`, then reads documents in
  stored order until the group's budget is reached. Whole documents are kept,
  so budgets can be exceeded slightly.

This approximately matches the released corpus's text proportions, not an exact
uniform document sample or necessarily the original model's training mixture.
Three shards per group is a small sample, and some estimates are noisy. Output
is grouped by topic; documents are not globally shuffled.

## Download

From this directory:

```bash
python download_dolma.py
```

`--output /absolute/path/dolma3_250gb.jsonl` writes to another directory. The
pinned estimates and revision are always read next to the downloader source.
For the combined CPU-only Beaker download and sample job, see
[`../scripts/README.md`](../scripts/README.md).

It needs about 4 CPUs and 32 GB RAM, no GPU, and roughly a day. It writes `dolma3_250gb.jsonl.part` and renames it to
`dolma3_250gb.jsonl` on success. Restarting scans the partial file to recover
progress, skips completed topics and saved records, then appends the remaining
data. Keep the topic estimates, revision, and seed unchanged when resuming.
Do not launch another download while one is running.

`estimate_dolma_text_sizes.py` and `dolma_sampled_shards.jsonl` retain the
estimation code and measurements; they are not needed for the download itself.

## Uniform 100 GB sample

```bash
python repeat_dolma.py --sample-fraction 0.4 --seed 42 --output dolma3_100gb.jsonl
```

Selects exactly 40% of the full file's records (rounded down), uniformly without
replacement, and preserves their source order and JSON contents. Every record
has the same inclusion probability. This yields approximately 100 GB of UTF-8
text; the exact text size is reported on completion, and JSON/metadata make the
file larger. Records are neither truncated nor repeated. Progress bars cover
counting and sampling. The script writes `.part`, checks that the source has not
changed, and renames on success; existing output/partial files are not overwritten.

## Repeated 100 GB samples

```bash
python repeat_dolma.py --input dolma3_100gb.jsonl --repeat 2 --seed 42
python repeat_dolma.py --input dolma3_100gb.jsonl --repeat 4 --seed 42
```

These create `dolma3_100gb_repeat_2.jsonl` and `dolma3_100gb_repeat_4.jsonl`.
Repeat-2 uniformly samples half the input records without replacement and writes
two identical passes; repeat-4 samples one quarter and writes four identical
passes. Each sample retains source order. Both outputs contain approximately
100 GB of UTF-8 text, with record counts rounded down to a multiple of the repeat
factor. The same progress bars, `.part` publishing, and overwrite checks apply.
