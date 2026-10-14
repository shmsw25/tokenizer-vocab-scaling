# Jupiter / Weka / Beaker pipeline

These scripts submit jobs in `ai2/flex2`. Model workflows run training,
CPU conversion, and single-GPU evaluation sequentially in **one shell script
and one Beaker allocation** (`run_model.sh`).
Download/sample jobs (including each smaller pool) can run on **any accessible cluster with Weka access**;
other CPU stages default to `ai2/phobos`, and model training uses H100 `ai2/jupiter`.
The active campaign now places unstarted CPU stages on `ai2/saturn` or `ai2/neptune`
at the user's request, because Phobos is full.
Downloading/sampling, tokenizer training, and tokenization explicitly request
**zero GPUs**. Beaker may reserve a GPU slot for a zero-GPU download on GPU
clusters; the campaign records this reservation, while CUDA remains disabled.
Model training defaults to **eight H100 GPUs on one node**. Submission is asynchronous:
wait for each prerequisite experiment to **succeed** before submitting its next
stage. No experiment is submitted by setup or by `--dry-run`.

## Adaptive study

The active campaign's `study` record extends the existing receipts in place.
Run `.venv/bin/python scripts/campaign.py --advance` every 30 minutes; never
reinitialize it. `adaptive_study.py` manages nine independent 5GB LR brackets:
baseline, half the LR, then another halving if validation BPB improves or twice
the baseline otherwise. The lowest final validation BPB selects the LR for
that model/vocabulary pair for 100GB. The October 14 override adds an independent
1e-3 / 2e-3 comparison for every model/vocabulary pair on both 10GB and 20GB,
recorded in `study.pool_lr_search`. Choose each pool's lowest final validation BPB;
keep held-out evaluation separate from LR selection. This supersedes transferring
one 5GB-selected rate to those two pools.
Completed matching runs are reused, and old pending recipes remain in state
as superseded plans. Existing running allocations are preserved.

Every new study job requests `minRuntime: 3h`. Admission counts all the owner's
live and queued Beaker jobs, including retained eight-GPU allocations during
conversion/evaluation. CPU requests still specify GPU 0; admission conservatively
budgets up to eight scheduler-reserved GPUs on Weka GPU clusters. Unknown
allocation state blocks new submissions. Other applications submitting jobs
concurrently are outside this controller; each admission rechecks live usage.

New training logs to W&B using the existing `SEWONM_WANDB_API_KEY` Beaker secret.
The monitor queries W&B summaries and retains Beaker loss logs. An in-job guard
stops immediately on non-finite CE. For finite loss, a 50-observation median must
remain above twice its best post-warmup value for 1,000 optimizer steps, with no
decrease in that median for 500 steps. Any decrease resets the recovery grace.
The guard starts after warmup plus 100 steps and permits recovery after spikes.
A stopped trial is not converted or counted as completed.

The user authorized resuming all 16 old guard stops at their original LRs.
`--resume-latest` is operational: it does not change model paths or the recipe.
It requires complete checkpoint metadata, optimizer shards, and every rank's
trainer state before restoring model, optimizer, loader, trainer, and RNG state.
`RESUME_VERIFIED` records the actual restored step. Old stop records and receipts
are archived; they must not mark the resumed attempt unhealthy. Use repeated
`--retry-node KEY` after live terminal-state checks. Resumes take admission
precedence over new LR variants; `--max-submissions 1` permits a canary rollout.

After selected-LR 10GB/20GB evaluations, the 100GB parent is copied once to
`raw_text/100GB/training/`, excluding the same 1,000 shared held-out records.
The immutable parent remains available. Train/tokenize 32K/64K/128K on that
eligible corpus. Each 100GB main run saves permanent full-state checkpoints at
`floor(epoch_steps * fraction) - round(floor(epoch_steps * fraction) * 0.1)`
for fractions 0.05, 0.1, 0.2, and 0.4. Independent branches restore each checkpoint
and decay to its horizon; the main run retains full LR until its own final 10%
decay. Each branch also chains conversion/evaluation, preserving its checkpoint.
The scheduler uses the branch's absolute horizon without repeating warmup.

After these finish, the conditional 8K/16K extension is triggered if tuned 32K
held-out BPB beats both larger vocabularies by at least 0.001 in at least five
of nine 5GB/10GB/20GB coordinates, including at least two models on 5GB. This
is a descriptive trigger, not a significance test. The extension trains new
tokenizers, tunes LR on 5GB, transfers it, and repeats the 100GB horizons.
The report records the decision, LR selection, all variants, and Brian's
supplied screenshot as a reference with different settings and token counts.

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

# Example override; still 128 global sequences per optimizer step:
scripts/submit_train_model.sh --data-size "$DATA_SIZE" --vocab-size "$VOCAB_SIZE" --model-size medium_large \
  --gpus 8 --micro-batch-size 8 --batch-size 128 --learning-rate 1e-3
```

Every run uses WSD: a fixed linear warmup, stable LR, then linear
decay (default last 10%) to zero. Warmup follows the infinite-compute recipe: `24 * 1024 = 24576` sequences,
which is exactly 192 optimizer steps at batch 128, independent of corpus size
and vocabulary. Warmup is not a fraction of the run. The trainer instantiates `olmo_core.optim.WSD`
directly; the launcher exposes no alternative scheduler. The context is always

```text
seq_length = round(4096 * tokens_per_byte(target) / tokens_per_byte(32000))
```

Model-training jobs explicitly set Beaker `context.minRuntime: 8h` for 32K vocabulary
and `0s` for other vocabularies. Override with `--min-runtime`, using an integer
duration in seconds, minutes, or hours (for example `0s`, `30m`, or `8h`).
The REST API receives the equivalent integer number of nanoseconds.
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

Repack is the default training data loader. The document-permutation and gathering
code comes from EMO `sewonm/icsl` at `6ef92fc3a45896a8d734ea4f852a3d35950e4abb`;
the rest of that branch is not imported. Epoch 1 uses the original FSL reader,
instance shuffle, and incomplete-sequence/global-batch dropping, so the current
single-epoch study is unchanged. From epoch 2, whole documents are deterministically
reshuffled using the upstream seed domain, then concatenated into new fixed-length
sequences. Source token files are never rewritten. Later-epoch resumes verify the
repack version and boundary checksum; old FSL epoch-1 checkpoints remain compatible.

New tokenization records exact spans in `train.documents.csv.gz` and
`eval.documents.csv.gz`, with checksums/counts in `repack.json`. Existing single-epoch
runs need no sidecars. Before using an existing token stream for **multi-epoch**
training, run this CPU-only command on a machine with the Weka mount:

```bash
python scripts/prepare_repack.py --data-dir \
  /oe-training-default/sewonm/tokenizer_project/tokenized_data/default_5GB_32000
```

This replays the saved tokenizer and train/eval split against the original raw pool,
compares every token, and publishes only document metadata. It never modifies
`train.bin`, `eval.bin`, or `meta.json`; failed `.part` metadata is retained for
inspection. An EOS scan is insufficient because EOS can occur inside a document.
Multi-epoch training fails before its first update if valid spans are unavailable.
For an explicit ordinary-loader ablation, set `training.data_loader` to `fsl` in
the model config, or pass `--data-loader fsl` directly to `train.py`. The effective
loader is recorded in checkpoint configs and training metrics. Existing model
path fingerprints and submitted immutable code snapshots remain unchanged.

Ratios are measured by tokenization over the same raw corpus, including one EOS
per document. The job validates corpus/split agreement and token file sizes,
then writes the resolved `launch_config.json` before invoking `torchrun`.
There is no sequence-length override in these submission scripts.

The default is one epoch and 128 global sequences per optimizer step on eight
H100 GPUs. Per-GPU microbatches are 16 (`small`), 16 (`medium`), and 8
(`medium_large`), with 1, 1, and 2 accumulation steps respectively. Gradient accumulation is computed as
`batch_size / (gpus * micro_batch_size)` and must be integral. The medium and medium_large microbatches come from the infinite-compute
project; small is capped at 16 so eight GPUs fit within global batch 128.
They are fixed across vocabulary sizes. Peak LR is 2e-3 for small and medium,
and 1e-3 for medium_large; weight decay remains 0.1.
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

The graph contains 115 logical stages, including 27 training, conversion, and
evaluation triples. New training jobs run each triple in one Beaker job through
`scripts/run_model.sh`: training must succeed before CPU conversion, which must
succeed before single-GPU evaluation. The eight-GPU allocation remains reserved
until the shell finishes. There is no hourly-monitor delay between these stages.
All model arguments are passed unchanged; conversion temporarily hides CUDA,
and evaluation sees only the first assigned GPU. Stage summaries are published
separately, so an evaluation failure does not erase successful training.

For a standalone chained submission, use the usual parameterized command with
`--chain-evaluation`, for example:

```bash
python scripts/submit.py train-model --data-size 5GB --vocab-size 64000 \
  --model-size small --batch-size 128 --learning-rate 2e-3 --chain-evaluation
```

Existing running training jobs keep their immutable scripts. Their follow-up can
use `convert-model --chain-evaluation --cluster ai2/jupiter` to perform CPU
conversion followed by evaluation in one allocation requesting one GPU. Already
running standalone conversion/evaluation jobs remain independent. The campaign
claims chained downstream stages before submission and never submits them again
as separate jobs. Completed training and valid conversion/evaluation artifacts
are reused on a retry.
Evaluation probes the shared prepared dataset cache offline before contacting
Hugging Face. BLiMP still requires all 67 subsets. Cache misses serialize Hub
access across jobs and honor rate-limit retry delays; scoring and benchmark
contents are unchanged. Retry an evaluation failure from its evaluation node
after its shared job stops, preserving successful training and conversion.
`--migrate-pools` upgrades the old unsubmitted 100GB plan while preserving its
current download receipt and history; it refuses to discard submitted downstream
jobs or migrate twice. Do not reinitialize an existing campaign.
It records intended names before submission and recovers uncertain
submission outcomes by looking up those names. No live training job is restarted
or duplicated. Failed jobs remain visible for diagnosis; after fixing the cause,
`--advance --retry-node NODE_KEY` explicitly retries a confirmed failure, resuming
existing Weka work where supported. It allows at most three failure-recovery
attempts per node; recorded user-requested recipe or scheduling changes do not
consume this retry allowance.

Each poll saves job status, observed GPU assignments, log tails, configurations,
result manifests, and current ETAs. Download ETA uses recent observed text-byte
throughput and excludes the subsequent sampling pass; model ETA comes from EMO.
Queue wait and unmeasured stages are reported as unknown. Install the local report
dependency with `python -m pip install -r requirements-report.txt` (included in
`setup_env.sh all`). The self-contained `.build/report.html` shows per-pool
tokenizer counts, three FLOPs plots with a metric selector, and a run grid grouped
by pool, K-formatted vocabulary, and model. Each LR/WD run shows its training state
and held-out Dolma BPB, BLiMP, ARC-E, HellaSwag, and PIQA together.
The compact grid reserves 25% of the table width for pool/vocabulary/model and
75% for runs. Each run is one line with scientific-notation LR (for example
`2e-03`), WD, state, and the five scores aligned beneath their headers.
The total-training-tokens table shows exact training-split counts and percentage
changes relative to the same pool's 32K tokenizer, before dropping the incomplete
final batch. Detailed job status and ETA stay in campaign state and hourly chat
updates rather than a separate report section.
FLOPs are calculated only after target and same-pool 32K tokenization finish:
complete global batches consumed in one epoch times the pinned EMO model's
forward/backward FLOPs per token at the scaled context. Plot points require
completed training and the selected metric. Multiple tuning runs select the
lowest final validation BPB, consistently across all plotted metrics.
Training emits loss observations without changing the optimization recipe.
Full hyperparameters, token counts, FLOPs, losses, job links, and provenance
remain in downloadable JSON. Completion requires all stages and all five
metrics for every evaluation.
The controller is a single polling command, not its own background scheduler;
the requested hourly chat heartbeat invokes it and reports progress.

## Resource and placement overrides

Every submission supports `--cpus`, `--memory`, `--workspace`, `--cluster`,
`--priority`, `--image`, and `--name`. Workspace and cluster can also default
from `BEAKER_WORKSPACE` and `BEAKER_CLUSTER`. Model training defaults to `urgent`
priority; other stages default to `normal`. Model-training jobs default to
`minRuntime: 8h` for 32K vocabulary and `0s` otherwise. Other stages are non-preemptible
unless `--preemptible` is supplied. CPU scripts expose
no GPU-count flag and always emit `gpuCount: 0`. GPU submissions discover and
check H100 availability and use the requested cluster as their placement
constraint, without a hostname list. The campaign uses `ai2/jupiter` for training;
verify the actual GPU assignment when each job starts. GPU clusters use
whole GPU slots even for `gpuCount: 0`. Download and raw sampling permit this scheduler
reservation to broaden placement; the monitor reports requested and assigned
counts separately. Other CPU stages default to Phobos; use
`--cluster ai2/saturn ai2/neptune --allow-gpu-reservation` to allow either of
those clusters while still requesting `gpuCount: 0` and disabling CUDA.
Campaign nodes retain `placement_clusters` and `allow_gpu_reservation`, so
retries and later CPU stages preserve their configured placement. Confirm an
old queued experiment is canceled before retrying it on the new placement;
its receipt remains in `previous_attempts`.
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

### Shared 100M-token validation and corrected medium_large (2026-10-14)

All three model configurations now use EMO's `TransformerBlockType.default`
(pre-norm RMSNorm) and `qk_norm=false`. The previous medium_large architecture
used reordered norms and QK normalization; its runs are preserved under
`models/medium_large_old` and excluded from report plots, grid, and LR selection.
Corrected medium_large runs start from scratch at the user-fixed LR `1e-03`,
with distinct `prenorm-v2` node keys. `study.fixed_learning_rates` supersedes
both its 5GB LR bracket and the larger-pool two-rate search, including later
100GB horizons and any smaller-vocabulary cohort. Small/medium LR searches
continue unchanged. Tokenizers and training streams are reused. Tied embeddings, embedding WD,
model dimensions, WSD, batch size, and the recovery-aware loss guard are retained.

`prepare-validation` creates `validation/dolma100m-v1/documents.jsonl` from the
complement of the original 100GB parent within the downloaded 250GB sample.
It reconstructs and hashes the original parent sampling, removes exact text
matches against that entire parent (including the 1000-document test sample),
shuffles documents deterministically, and stops at a whole-document boundary
at or above 100M tokens under the frozen 5GB/32K tokenizer. It publishes the raw
text hash, source hashes, text bytes, document counts and target overshoot.
All vocabularies and pools use exactly those documents, so their token counts
naturally differ. This matches the scale of arXiv:2610.15965's 100M-token DCLM
validation, not its dataset or tokenizer. Existing training files are unchanged.

The CPU preparation job also writes the initial nine validation streams.
Future vocabularies/pools use the CPU `tokenize-validation` stage. New training
defaults to `--validation-id dolma100m-v1`; this provenance participates in the
run fingerprint. Already-submitted/resuming jobs keep their original
configuration, with explicit `--validation-id legacy` on future retries to
preserve the exact path and checkpoint configuration. This flag preserves the
training recipe only: final Dolma evaluation always uses the shared corpus.

The one-time backfill uses one-GPU `validate-model` jobs after each existing
checkpoint's conversion/evaluation completes. They replace the current
`evaluation/dolma_bpb.json` and the Dolma entry in `evaluation_summary.json`,
preserving other benchmarks and keeping old scores only in audit history.
The controller tracks completion in `study.validation_backfill` and deletes
the one-time local migration script after all its targets are scored (or
explicitly excluded as unhealthy, incomplete trials). The reusable scorer is
part of the permanent pipeline. New training shells automatically score the
same corpus during `evaluate-model` inside the original allocation; they do
not submit an extra validation job. LR selection waits for common scores,
rather than mixing them with the earlier pool-specific 1% split.
The post-training scorer predicts every token except the first in fixed context
windows and applies the repository's token-CE-to-byte normalization. The report
replaces the old 1000-document Dolma score with this shared validation BPB and
shows blanks until each replacement is ready. This corpus now supplies both
the reported Dolma metric and LR selection; it is not an independent test set.
