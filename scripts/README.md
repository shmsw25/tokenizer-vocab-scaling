# Jupiter / Weka / Beaker pipeline

These scripts submit **one stage per Beaker experiment** to `ai2/jupiter` in
`ai2/flex2`. Downloading/sampling, tokenizer training, and tokenization request
**zero GPUs**. Model training defaults to **one GPU**. Submission is asynchronous:
wait for each prerequisite experiment to **succeed** before submitting its next
stage. No experiment is submitted by setup or by `--dry-run`.

## Local setup

The repository contains the complete pinned EMO source with the upstream
`tied embeddings + DDP gradient accumulation` patch already applied. It is
ordinary tracked source, not a Git submodule. See `../EMO_VENDOR.json` and
`../EMO/LICENSE`. Setup installs it as an editable local package and verifies
the patched files.

```bash
# From the repository root, in Python 3.12:
python3.12 -m venv .venv
source .venv/bin/activate
./setup_env.sh
```

For submission alone, Python 3.10+ and
`python -m pip install -r scripts/requirements-launch.txt` suffice. The wrappers
use `.venv/bin/python` if present, or `python3`; set `PYTHON` to override it.
Beaker authentication uses the existing `~/.beaker/config.yml`, or the
`BEAKER_TOKEN` environment variable. The Beaker CLI is not required. Never
commit tokens. The dataset is public; `--hf-token-secret SECRET_NAME` is an
optional way to supply a workspace Beaker secret as `HF_TOKEN` to a job.

CPU jobs use the `python:3.12-slim-bookworm` Docker image and install only
`scripts/requirements-cpu.txt`, with no PyTorch or CUDA dependency. Training
uses Beaker image `01KAXVQ0YN242MQ9YEQDQQK512`
(`tylerr/olmo-core-tch280cu128-2025-11-25`) plus the local patched EMO package.
Job environments are isolated in an ephemeral virtual environment; large data
and outputs persist on Weka. Image/dependency installation requires outbound
network access from the job.

## Paths

Weka bucket `oe-training-default` is mounted at `/oe-training-default`, so every
persistent data, tokenizer, and model artifact lives below
`/oe-training-default/sewonm/tokenizer_project` (singular). Scripts create the
root and required subdirectories when the first job runs.

```text
tokenizer_project/
  raw_text/250GB/dolma3_250gb.jsonl
  raw_text/100GB/dolma3_100gb.jsonl
  tokenizers/default_100GB_<vocab_size>/tokenizer.json
  tokenized_data/default_100GB_<vocab_size>/
    train.bin
    eval.bin
    tokenizer.json
    meta.json
    pipeline.json
  models/<model_size>/default_100GB_<vocab_size>/
    wsd_lr<lr>_wd<wd>_bs<batch>_seq<length>_<budget>_seed<seed>_wu<warmup>_decay<decay>_<hash>/
```

All vocabulary-dependent stages default to `--data-size 100GB`. The optional
`--data-size 250GB` switches their inputs and output names together; it requires
a 32k reference tokenized from that same 250GB corpus. `--root` may override the
project directory, but Beaker submissions require it to stay inside the Weka
mount. Model config names currently available are `small`, `medium`, and
`olmo2_1b`. A new size such as `medium_large` can be added as
`configs/medium_large.json` without editing any script.

## Review a job before launching

```bash
scripts/submit_download_data.sh --dry-run
scripts/submit_train_tokenizer.sh --vocab-size 48000 --dry-run
scripts/submit_tokenize_data.sh --vocab-size 48000 --dry-run
scripts/submit_train_model.sh --vocab-size 48000 --model-size small --dry-run
```

`--dry-run` prints the complete JSON experiment spec without reading credentials,
uploading code, creating Weka directories, downloading data, or submitting jobs.
The code dataset field is a placeholder until real submission. Add
`--spec-file .build/job.json` to save a copy. Non-dry-run submissions require a
clean, committed working tree. They upload `git archive HEAD` and the committed
bootstrap script to a new immutable Beaker dataset, then launch from it. Thus
EMO travels with the job, and pushing to GitHub is not required to submit.
Keep the code dataset for reproducibility. Each submission prints the experiment
URL and code dataset ID.

## 1. Download 250GB and sample 100GB (CPU only)

```bash
scripts/submit_download_data.sh
```

This runs the original pinned Dolma downloader, then
`repeat_dolma.py --sample-fraction 0.4 --seed 42`. It uniformly selects exactly
40% of records without replacement and writes each selected record once. There
is **no repetition**. The names refer to approximate UTF-8 text size: JSON and
metadata require more disk space, and the uniform sample is not an exact byte
quota. Both raw versions are retained, so allow more than 350GB of raw storage,
plus tokenized streams and checkpoints. See `../data/README.md` for the source
mixture, pinned Hugging Face revision, and sampling method.

The job defaults to 4 CPUs and 32 GiB RAM. A complete download/sample is reused.
Downloading can resume a valid `.part` file. Sampling refuses an existing `.part`
file; after a failed sampling attempt, inspect it and move it aside before
rerunning. A truncated download line also requires inspection before resuming.
A filesystem lock prevents concurrent download/sample writers.

## 2. Build the 32k reference (CPU only)

```bash
scripts/submit_train_tokenizer.sh --vocab-size 32000
# After tokenizer training succeeds:
scripts/submit_tokenize_data.sh --vocab-size 32000
```

Both use the uniform 100GB file. The reference is required even when you only
plan to train a model with a different vocabulary size. Model training does not
train or tokenize a missing reference inside a GPU allocation.

## 3. Build any target vocabulary (CPU only)

```bash
VOCAB_SIZE=48000
scripts/submit_train_tokenizer.sh --vocab-size "$VOCAB_SIZE"
# After tokenizer training succeeds:
scripts/submit_tokenize_data.sh --vocab-size "$VOCAB_SIZE"
```

Vocabulary size is an integer argument, not a hardcoded list. The only special
size is the required 32000-token context reference. If the target is 32000,
step 2 already produced these artifacts.

Tokenizer training requests 64 CPUs and 128 GiB RAM; tokenization requests 16
CPUs and 64 GiB RAM. `RAYON_NUM_THREADS` follows the requested CPU count. The
split is fixed at seed 42 and 1% evaluation documents for all vocabularies.
Artifacts store raw-file identity and tokenizer hashes, and completed stages
are reused only when their provenance agrees. A tokenization attempt without
`meta.json` is incomplete and restarts from the beginning under a writer lock.
Do not modify published raw files, tokenizers, or tokenized streams in place.

## 4. Train a model (GPU)

After target and 32k tokenization both succeed:

```bash
scripts/submit_train_model.sh --vocab-size "$VOCAB_SIZE" --model-size small

# Example override; still 128 global sequences per optimizer step:
scripts/submit_train_model.sh --vocab-size "$VOCAB_SIZE" --model-size olmo2_1b \
  --gpus 2 --micro-batch-size 1 --batch-size 128 --learning-rate 1e-4
```

Every run uses WSD: linear warmup (config default 3%), stable LR, then linear
decay (default last 10%) to zero. The trainer instantiates `olmo_core.optim.WSD`
directly; the launcher exposes no alternative scheduler. The context is always

```text
seq_length = round(4096 * tokens_per_byte(target) / tokens_per_byte(32000))
```

Ratios are measured by tokenization over the same raw corpus, including one EOS
per document. The job validates corpus/split agreement and token file sizes,
then writes the resolved `launch_config.json` before invoking `torchrun`.
There is no sequence-length override in these submission scripts.

The default is one epoch, 128 global sequences per optimizer step, and one
sequence per GPU microbatch. Gradient accumulation is computed as
`batch_size / (gpus * micro_batch_size)` and must be integral. A small fixed
microbatch avoids vocabulary-specific tuning; increase it if memory allows.
Per-GPU memory still depends on model and vocabulary size, so arbitrary large
values are not guaranteed to fit. LR and weight decay default to the selected
model config. Other controls are `--learning-rate`, `--weight-decay`, `--epochs`,
`--seed`, `--max-steps`, and `--no-compile`. `--max-steps` gives an optional short
run with its own WSD horizon/output path; omit it for the full epoch budget.

Readable hyperparameters and a hash of the complete resolved configuration
form the model directory name. Repeating the same command resumes EMO's latest
checkpoint there. Changed LR/batch/schedule/configuration produces a different
path. A writer lock prevents simultaneous training into the same directory.
The exact submitted commit is recorded in Beaker and `source_revisions.txt`.

## Resource and placement overrides

Every submission supports `--cpus`, `--memory`, `--workspace`, `--cluster`,
`--priority`, `--image`, and `--name`. Workspace and cluster can also default
from `BEAKER_WORKSPACE` and `BEAKER_CLUSTER`. Priority defaults to `normal`;
jobs are non-preemptible unless `--preemptible` is supplied. CPU scripts expose
no GPU-count flag and always emit `gpuCount: 0`. GPU training is single-node;
`--gpus` is GPUs on that node.

## Validation

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/verify_emo.py
.venv/bin/python -m pip check
```

Tests run offline: Beaker schema/resource checks, uniform nonrepeated sampling,
actual tokenizer training/tokenization at two small vocab sizes, identical
splits, provenance checks, context scaling, output naming, WSD values, and tied
embedding identity. On the local macOS Conda Python, `readline` crashes during
pytest capture initialization; `-p no:capture` avoids that interpreter issue.
GPU execution and the full 250GB download need a subsequent Beaker run; neither
is performed by the validation tests.
