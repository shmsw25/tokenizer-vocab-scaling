# Tokenizer vocabulary scaling

Train the same language model with BPE tokenizers of different vocabulary sizes
(32k, 64k, 128k) on Dolma 3 and compare them in bits per byte.

The pipeline has four steps, one script each:

| Step | Script | Input | Output |
|---|---|---|---|
| 1. Train a tokenizer | `train_tokenizer.py` | raw text | `tokenizer_32k.json` |
| 2. Tokenize the data | `tokenize_data.py` | raw text + tokenizer | a directory of token files |
| 3. Train a model | `train.py` | that directory + a config | checkpoints |
| 4. Evaluate | `eval/` | a checkpoint | JSON metrics |

Raw text is a `.jsonl` file, or a directory of `.jsonl` / `.jsonl.gz` / `.jsonl.zst`
files, where every line is `{"text": "one document", ...}`.

## Setup

For the Jupiter/Weka workflow, see **[scripts/README.md](scripts/README.md)**.
It reuses a uniform 100GB parent to create six independent smaller pools,
trains 18 tokenizers, and submits 27 one-epoch model runs on 5GB/10GB/20GB.
CPU data/tokenizer jobs and GPU model jobs run separately in Beaker; context
length always matches the selected pool's 32k tokenizer.

```bash
conda create -n tokenizer-scaling python=3.12 && conda activate tokenizer-scaling
./setup_env.sh
```

`setup_env.sh` installs `requirements.txt`, the Beaker submission SDK, and the
local [EMO](https://github.com/allenai/EMO) training library. `EMO/` is checked
into this repository at pinned commit `246fdec72e599319c717aefe29a1d1fdad7c97a0`,
with `emo_tied_embeddings.patch` already applied. The patch adds tied
input/output embeddings and makes gradient accumulation sync gradients once
per optimizer step under DDP. `emo_optional_te.patch` lets the selected PyTorch
attention backend load when optional Transformer Engine CUDA libraries are unavailable;
explicit Transformer Engine selection still fails. `emo_repack.patch` ports document
repacking from `sewonm/icsl`: epoch 1 is unchanged; later epochs reshuffle whole
documents before forming sequences. Repack is the default training loader.
`EMO_VENDOR.json` records provenance and hashes;
setup verifies the patched files. No submodule or extra clone is needed.
Use `./setup_env.sh cpu` for the CPU-only data/tokenizer dependencies.

## 1. Train tokenizers

```bash
for v in 32000 64000 128000; do
    python train_tokenizer.py --data /path/to/raw --vocab-size $v --output tokenizers/tokenizer_$v.json
done
```

Byte-level BPE with `<|endoftext|>` as token 0. Training on 100 GB of text used
64 CPUs and 128 GB RAM; set `RAYON_NUM_THREADS` to the
number of CPUs.

## 2. Tokenize the data

```bash
for v in 32000 64000 128000; do
    python tokenize_data.py --data /path/to/raw --tokenizer tokenizers/tokenizer_$v.json --output-dir tokenized/$v
done
```

Each output directory holds:

- `train.bin`, `eval.bin`: flat `uint32` token streams, each document followed by EOS.
- `tokenizer.json`: a copy of the tokenizer.
- `meta.json`: document, byte and token counts, and `tokens_per_byte`.

1% of documents go to `eval.bin`. The split depends only on the seed and the
document order, so all tokenizers see the same train and eval documents.

## 3. Train

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/small.json --data-dir tokenized/32000 --output-dir runs/small_32k
```

The config holds the model and optimizer settings. Vocabulary size and tokenizer
come from `--data-dir`, so one config serves every vocabulary size. Rerunning the
same command resumes from the last checkpoint in `--output-dir`.

| Config | Layers | Width | Parameters with 32k / 64k / 128k | Learning rate |
|---|---:|---:|---|---:|
| `configs/small.json` | 12 | 512 | about 67M / 83M / 116M | 2e-3 |
| `configs/medium.json` | 16 | 1024 | about 301M / 334M / 399M | 2e-3 |
| `configs/medium_large.json` | 16 | 2048 | 1,139M / 1,205M / 1,336M | 1e-3 |

The fixed warmup matches the infinite-compute project: 24,576 sequences, or
192 steps at global batch 128. Warmup, decay, and epoch/step budget are recorded
in the config and omitted from the readable output directory name.

All configs use tied embeddings, sequence length 4096, AdamW with weight decay
0.1, 192-step linear warm-up, constant learning rate, linear decay to zero over the
last 10% of steps, BF16 autocast, and one epoch.

### Batch size

Sequences per optimizer step = `micro_batch_size` x number of GPUs x
`gradient_accumulation_steps`. The configs give 128 sequences per step on eight
GPUs, which is 524,288 tokens at sequence length 4096. Keep that product at 128
when changing the GPU count:

```bash
# four GPUs, small model
torchrun --standalone --nproc_per_node=4 train.py --config configs/small.json \
    --micro-batch-size 16 --gradient-accumulation-steps 2 ...
```

`micro_batch_size` is what has to fit in GPU memory. On 141 GB H200 GPUs the
1B model used 2 with the 32k and 64k vocabularies and 1 with 128k (pass
`--micro-batch-size 1 --gradient-accumulation-steps 64`).

### Sequence length

By default every vocabulary size uses the config's sequence length in tokens,
which is how the original runs were done. A larger vocabulary then sees more
text per sequence. To give every model the same amount of text per sequence
instead, name the reference tokenizer's directory:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/small.json --data-dir tokenized/128000 --output-dir runs/small_128k_matched \
    --match-context-of tokenized/32000
```

This sets `seq_length = 4096 x tokens_per_byte(128k) / tokens_per_byte(32k)`,
rounded. The batch stays at 128 sequences per step, so every model also sees the
same number of bytes per step. `--seq-length N` sets it by hand. The value used
is printed at start-up and saved in each checkpoint's `config.json`.

### Logging

Set `"report_to": "wandb"` in the config to log to Weights & Biases (run
`wandb login` first). The default is no external logging; losses print to stdout.

## 4. Evaluate

See [`eval/README.md`](eval/README.md). In short:

```bash
python eval/checkpoint_converter.py runs/small_32k/step10000
python eval/eval_suite.py eval/model_checkpoints/small_32k/step10000 blimp,arc_easy,hellaswag,piqa,dolma_bpb
```

## Data

`data/` has the scripts that built the original corpus: a 250 GB sample of
`allenai/dolma3_mix-6T` proportional to the release's source mix, and a uniform
100 GB subsample of it used for all runs. See [`data/README.md`](data/README.md).
No data is included in this repository.

## What changed from the original runs

The original trainer tokenized the data itself on first launch and read the
tokenizer and data paths from the config. Here tokenization is its own script
and the trainer reads a tokenized directory. The model, optimizer, schedule,
data split and token file format are unchanged. `--match-context-of` is new
and was not used in the original runs.
