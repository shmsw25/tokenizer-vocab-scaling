# Jupiter / Weka / Beaker pipeline

These scripts submit **one stage per Beaker experiment** in `ai2/flex2`.
Download/sample jobs (including each smaller pool) can run on **any accessible cluster with Weka access**;
other CPU stages use `ai2/phobos`, and model training uses H100 `ai2/jupiter`.
Downloading/sampling, tokenizer training, and tokenization explicitly request
**zero GPUs**. Beaker may reserve a GPU slot for a zero-GPU download on GPU
clusters; the campaign records this reservation, while CUDA remains disabled.
Model training defaults to **eight H100 GPUs on one node**. Submission is asynchronous:
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
`python -m pip install -r requirements-launch.txt` suffice. The wrappers
use `.venv/bin/python` if present, or `python3`; set `PYTHON` to override it.
Beaker authentication uses the existing `~/.beaker/config.yml`, or the
`BEAKER_TOKEN` environment variable. The Beaker CLI is not required. Never
commit tokens. The dataset is public; `--hf-token-secret SECRET_NAME` is an
optional way to supply a workspace Beaker secret as `HF_TOKEN` to a job.

CPU data jobs use the `python:3.12-slim-bookworm` Docker image and install only
`requirements-cpu.txt`, with no PyTorch or CUDA dependency. Training
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
  raw_text/<data_size>/dolma3_<data_size_lowercase>.jsonl
  raw_text/<data_size>/pool.json
  raw_text/heldout/dolma3_100gb_shared_1000.json
  tokenizers/default_<data_size>_<vocab_size>/tokenizer.json
  tokenized_data/default_<data_size>_<vocab_size>/
    train.bin
    eval.bin
    tokenizer.json
    meta.json
    pipeline.json
  models/<model_size>/default_<data_size>_<vocab_size>/
    wsd_lr<lr>_wd<wd>_bs<batch>_seq<length>_seed<seed>_wt_EmbedWD_<hash>/
```

The campaign trains BPE vocabularies 32000/64000/128000 separately on **2.5GB,
5GB, 10GB, 20GB, 40GB, and 80GB** (18 tokenizers). It trains the three model
sizes only on **5GB, 10GB, and 20GB** (27 one-epoch models).
Pass `--data-size` to select a pool; standalone commands retain the legacy
100GB default. Every model requires its own pool's 32k reference tokenization.
`--root` may override the
project directory, but Beaker submissions require it to stay inside the Weka
mount. Model config names currently available are `small`, `medium`, and
`medium_large`. The campaign accepts model sizes and vocabulary sizes as arguments;
there are no per-vocabulary shell scripts. All requirements files live at the repository root.

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

The job defaults to 4 CPUs and 32 GiB RAM, with explicit `gpuCount: 0`.
Beaker requires a placement constraint when mounting Weka, so submission discovers
all clusters tagged `storage:weka`, filters user/workspace-budget restrictions and session-only clusters,
and allows the complete eligible list. If Beaker rejects a cluster because the
workspace lacks a required allocation or the cluster disallows batch tasks, that cluster is removed before resubmitting
the rejected request. This does not retry ambiguous outcomes or create duplicate jobs.
There is no single-cluster default.
Offline `--dry-run` uses an explicit placeholder for this dynamically resolved list.
Use `--cluster` only to restrict placement explicitly.
A complete download/sample is reused.
Downloading can resume a valid `.part` file. Sampling refuses an existing `.part`
file; after a failed sampling attempt, inspect it and move it aside before
rerunning. A truncated download line also requires inspection before resuming.
A filesystem lock prevents concurrent download/sample writers.

## 2. Create independent smaller pools (CPU only)

After the current 100GB job succeeds, submit separate jobs using the same
parameterized command:

```bash
for DATA_SIZE in 2.5GB 5GB 10GB 20GB 40GB 80GB; do
  python scripts/submit.py sample-pool --data-size "$DATA_SIZE"
done
```

The campaign does this automatically. Each job explicitly requests `gpuCount: 0`,
4 CPUs and 32 GiB RAM, and uses any eligible Weka cluster. The parent 100GB file
is reused without another download. Each size uses a distinct seed derived from
the size and base seed 42; these are **independent samples, not nested samples**.
Each sample selects exactly `floor(eligible_records * size_GB / 100)` records
uniformly without replacement, preserving source order. GB sizes are approximate
UTF-8 text sizes, not JSONL disk bytes or promised token budgets. The manifest
records exact document counts, text bytes, source/output hashes and seeds.
Token counts are measured later for the three model-training pools.

All six jobs exclude the same 1,000 randomly selected parent record indices
(fixed seed 20261012) and publish the same shared Dolma evaluation set. This
keeps evaluation records outside every smaller pool, including tokenizer
training. Independent pools can overlap with one another. Sampling uses bounded
memory and `.part` outputs; partial files are preserved and require inspection
before retrying. Completed pools are reused only with matching provenance.
All six sampling jobs must succeed before any of the 18 tokenizer jobs starts.

## 3. Build each pool's 32k reference (CPU only)

```bash
DATA_SIZE=5GB
scripts/submit_train_tokenizer.sh --data-size "$DATA_SIZE" --vocab-size 32000
# After tokenizer training succeeds:
scripts/submit_tokenize_data.sh --data-size "$DATA_SIZE" --vocab-size 32000
```

Both use the selected pool. Its reference is required even when you only
plan to train a model with a different vocabulary size. Model training does not
train or tokenize a missing reference inside a GPU allocation.

## 4. Build any target vocabulary (CPU only)

```bash
VOCAB_SIZE=48000
scripts/submit_train_tokenizer.sh --data-size "$DATA_SIZE" --vocab-size "$VOCAB_SIZE"
# After tokenizer training succeeds:
scripts/submit_tokenize_data.sh --data-size "$DATA_SIZE" --vocab-size "$VOCAB_SIZE"
```

Vocabulary size is an integer argument, not a hardcoded list. The only special
size is the required 32000-token context reference. If the target is 32000,
the reference commands already produced these artifacts. Only 5GB, 10GB, and
20GB need tokenized model-training streams in the current campaign.

Tokenizer training requests 64 CPUs and 128 GiB RAM; tokenization requests 16
CPUs and 64 GiB RAM. `RAYON_NUM_THREADS` follows the requested CPU count. The
split is fixed at seed 42 and 1% evaluation documents for all vocabularies.
Artifacts store raw-file identity and tokenizer hashes, and completed stages
are reused only when their provenance agrees. A tokenization attempt without
`meta.json` is incomplete and restarts from the beginning under a writer lock.
Do not modify published raw files, tokenizers, or tokenized streams in place.

## 5. Train a model (GPU)

After target and 32k tokenization both succeed:

```bash
scripts/submit_train_model.sh --data-size "$DATA_SIZE" --vocab-size "$VOCAB_SIZE" --model-size small

# Example override; still 256 global sequences per optimizer step:
scripts/submit_train_model.sh --data-size "$DATA_SIZE" --vocab-size "$VOCAB_SIZE" --model-size medium_large \
  --gpus 8 --micro-batch-size 8 --batch-size 256 --learning-rate 1e-4
```

Every run uses WSD: a fixed linear warmup, stable LR, then linear
decay (default last 10%) to zero. Warmup follows the infinite-compute recipe: `24 * 1024 = 24576` sequences,
which is exactly 96 optimizer steps at batch 256, independent of corpus size
and vocabulary. Warmup is not a fraction of the run. The trainer instantiates `olmo_core.optim.WSD`
directly; the launcher exposes no alternative scheduler. The context is always

```text
seq_length = round(4096 * tokens_per_byte(target) / tokens_per_byte(32000))
```

Every model-training job explicitly sets Beaker `context.minRuntime: 0s`.
All three model sizes currently tie input embeddings to the output projection
(`wt`). AdamW applies the configured weight decay to every trainable parameter,
including that shared matrix (`EmbedWD` for positive WD).
Each tag is omitted when its corresponding setting is disabled. Training,
conversion, and evaluation use the same directory naming.

The trainer retains a **full-state pre-decay checkpoint** after step
`total_steps - round(total_steps * decay_fraction)`, the last full-LR update.
It writes this permanent checkpoint synchronously before the first reduced-LR
update, retaining model weights, optimizer, trainer/data-loader position, and RNG
state for later multi-epoch continuation. `pre_decay_checkpoint.json` records
its path and boundary; the final checkpoint remains the evaluation checkpoint.
Very short smoke runs without a full-LR/decay transition record `not_applicable`.
Continuation from this branch point into a longer schedule is a separate run;
repeating the original command resumes its latest checkpoint as usual.

Ratios are measured by tokenization over the same raw corpus, including one EOS
per document. The job validates corpus/split agreement and token file sizes,
then writes the resolved `launch_config.json` before invoking `torchrun`.
There is no sequence-length override in these submission scripts.

The default is one epoch and 256 global sequences per optimizer step on eight
H100 GPUs. Per-GPU microbatches are 32 (`small`), 16 (`medium`), and 8
(`medium_large`), with 1, 2, and 4 accumulation steps respectively. Gradient accumulation is computed as
`batch_size / (gpus * micro_batch_size)` and must be integral. These settings come from the corresponding model sizes in the infinite-compute
project; they are fixed across vocabulary sizes.
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

## 6. Convert checkpoints and evaluate

```bash
# After training succeeds; pass the same model/training arguments to identify its output:
python scripts/submit.py convert-model --data-size "$DATA_SIZE" --vocab-size "$VOCAB_SIZE" --model-size small
# After conversion succeeds:
python scripts/submit.py evaluate-model --data-size "$DATA_SIZE" --vocab-size "$VOCAB_SIZE" --model-size small
```

Conversion requests **zero GPUs on Phobos**, uses the training dependency image,
exports the final completed checkpoint to Hugging Face format, and verifies
numerical agreement with EMO before publishing it. It verifies the shared
1,000-document held-out Dolma sample reserved during pool sampling. Evaluation requests
**one H100** and runs BLiMP, ARC-Easy, HellaSwag, PIQA, and Dolma BPB. The repository
uses byte-normalized downstream scoring, including BLiMP, so these accuracy
numbers should not be conflated with standard benchmark implementations.
Conversion outputs and resumable per-task evaluation JSON stay inside the
model directory. Small manifests are also exported through Beaker `/results/summary.json`.

## Run and monitor the campaign

The controller accepts `--pool-sizes`, `--training-pools`, arbitrary vocabularies
(including the required 32000 reference), and the three model sizes. It saves its state and HTML report under
ignored `.build/`, so monitoring does not dirty source snapshots. Initialize
**once**; adopting a receipt reuses an existing download instead of starting another.

```bash
python scripts/campaign.py --init --download-receipt .build/download-receipt.json \
  --vocabs 32000 64000 128000 --models small medium medium_large

# Repeat each hour; only successful prerequisites allow downstream submissions:
python scripts/campaign.py --advance

# Inspect/render saved state without network access or job submission:
python scripts/campaign.py --offline
```

The graph contains one original download/sample job, six independent pool
sampling jobs, 18 tokenizer jobs, nine tokenization jobs, 27 one-epoch training
jobs, 27 CPU conversions, and 27 evaluations (115 stages).
`--migrate-pools` upgrades the old unsubmitted 100GB plan while preserving its
current download receipt and history; it refuses to discard submitted downstream
jobs or migrate twice. Do not reinitialize an existing campaign.
It records intended names before submission and recovers uncertain
submission outcomes by looking up those names. No live training job is restarted
or duplicated. Failed jobs remain visible for diagnosis; after fixing the cause,
`--advance --retry-node NODE_KEY` explicitly retries a confirmed failure, resuming
existing Weka work where supported. It allows at most three attempts per node.

Each poll saves job status, observed GPU assignments, log tails, configurations,
result manifests, and current ETAs. Download ETA uses recent observed text-byte
throughput and excludes the subsequent sampling pass; model ETA comes from EMO.
Queue wait and unmeasured stages are reported as unknown. The HTML report at
`.build/report.html` is marked complete only after all stages succeed and all
27 evaluations have all five metrics. Results include pool size and a separate
18-tokenizer status table. It embeds downloadable JSON provenance.
The controller is a single polling command, not its own background scheduler;
the requested hourly chat heartbeat invokes it and reports progress.

## Resource and placement overrides

Every submission supports `--cpus`, `--memory`, `--workspace`, `--cluster`,
`--priority`, `--image`, and `--name`. Workspace and cluster can also default
from `BEAKER_WORKSPACE` and `BEAKER_CLUSTER`. Priority defaults to `normal`;
model-training jobs always use `minRuntime: 0s`. Other stages are non-preemptible
unless `--preemptible` is supplied. CPU scripts expose
no GPU-count flag and always emit `gpuCount: 0`. GPU submissions discover and
constrain placement to H100 nodes in the requested cluster. GPU clusters use
whole GPU slots even for `gpuCount: 0`. Download and raw sampling permit this scheduler
reservation to broaden placement; the monitor reports requested and assigned
counts separately. Other CPU stages use Phobos and reject GPU-cluster overrides.
GPU training is single-node;
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

The readable model path omits warmup, decay, and training budget (epoch/step
limit). All remain recorded in the configuration; its hash distinguishes
otherwise conflicting runs. `medium_large` is the renamed original 1B model,
with no architecture change.
