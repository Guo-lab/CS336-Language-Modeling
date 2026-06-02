# Python Scripts

Run these from the `assignment1-basics/` directory.

## `train_bpe_experiment.py`

Train experiment tokenizers. The script refuses to write into a non-empty `--out-dir` by default, so existing artifacts are not overwritten accidentally.

Downscaled TinyStories validation run with profiling:

```bash
.venv/bin/python scripts/train_bpe_experiment.py --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000 --special-token '<|endoftext|>' --out-dir artifacts/tokenizer_experiments/training/tokenizers_unoptimized/tinystories_valid_10k --profile --monitor-interval 30
```

Full TinyStories run:

```bash
.venv/bin/python scripts/train_bpe_experiment.py --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --special-token '<|endoftext|>' --out-dir artifacts/tokenizer_experiments/training/tokenizers_unoptimized/tinystories_train_10k --profile --monitor-interval 30
```

Full OpenWebText sample run:

```bash
.venv/bin/python scripts/train_bpe_experiment.py --input data/owt_train.txt --vocab-size 32000 --special-token '<|endoftext|>' --out-dir artifacts/tokenizer_experiments/training/tokenizers_unoptimized/owt_train_32k --profile --monitor-interval 30
```

## `tokenizer_experiments.py`

Run tokenizer compression and throughput experiments. By default this samples TinyStories and OpenWebText validation documents, loads the TinyStories 10K and OpenWebText 32K tokenizers, and evaluates the configured tokenizer/dataset combinations.

```bash
.venv/bin/python scripts/tokenizer_experiments.py --out-dir artifacts/tokenizer_experiments/evaluation/tokenizer_experiments
```

## `tokenize_dataset.py`

Convert raw text into a tokenized 1D `.npy` array for LM training. The output
can be loaded by `np.load(..., mmap_mode="r")`.

TinyStories examples:

```bash
.venv/bin/python scripts/tokenize_dataset.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
  --output artifacts/lm_data/tinystories_train_10k.npy \
  --dtype uint16
```

```bash
.venv/bin/python scripts/tokenize_dataset.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
  --output artifacts/lm_data/tinystories_valid_10k.npy \
  --dtype uint16
```

Inspect tokenized LM data and decode previews:

```bash
.venv/bin/python sanity_check/test_lm_data.py
```

## `transformer_accounting.py`

Print Transformer LM parameter and FLOPs accounting tables in the terminal.

```bash
.venv/bin/python scripts/transformer_accounting.py
```

## `sgd_lr_tuning.py`

Run the toy SGD learning-rate tuning experiment from Section 4.2.1.

```bash
.venv/bin/python scripts/sgd_lr_tuning.py
```

Change the tested learning rates or number of steps:

```bash
.venv/bin/python scripts/sgd_lr_tuning.py --lrs 10 100 1000 --steps 10
```

## `adamw_accounting.py`

Print AdamW training memory/FLOPs accounting for the GPT-2 XL-shaped model.

```bash
.venv/bin/python scripts/adamw_accounting.py
```

## `train_lm.py`

Train a Transformer language model from tokenized `.npy` data. The script
memory-maps train/valid arrays, logs metrics and samples, saves checkpoints,
and can resume from a saved checkpoint.

```bash
.venv/bin/python scripts/train_lm.py \
  --train-data artifacts/lm_data/tinystories_train_10k.npy \
  --valid-data artifacts/lm_data/tinystories_valid_10k.npy \
  --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
  --vocab-size 10000 \
  --run-name tinystories_smoke \
  --context-length 64 \
  --batch-size 16 \
  --num-layers 2 \
  --d-model 128 \
  --num-heads 4 \
  --d-ff 384 \
  --max-iters 500 \
  --warmup-iters 50 \
  --cosine-cycle-iters 500 \
  --log-every 10 \
  --eval-every 100 \
  --save-every 250 \
  --sample-every 100 \
  --sample-prompt "Once upon a time"
```

Enable W&B logging:

```bash
.venv/bin/python scripts/train_lm.py ... --wandb-project cs336-a1
```

Check the W&B code path without uploading:

```bash
.venv/bin/python scripts/train_lm.py ... --wandb-project cs336-a1 --wandb-mode disabled
```

Resume from a checkpoint by rerunning the same model/data command and adding
`--resume-from`:

```bash
.venv/bin/python scripts/train_lm.py \
  --train-data artifacts/lm_data/tinystories_train_10k.npy \
  --valid-data artifacts/lm_data/tinystories_valid_10k.npy \
  --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
  --vocab-size 10000 \
  --run-name tinystories_smoke \
  --context-length 64 \
  --batch-size 16 \
  --num-layers 2 \
  --d-model 128 \
  --num-heads 4 \
  --d-ff 384 \
  --max-iters 1000 \
  --warmup-iters 50 \
  --cosine-cycle-iters 1000 \
  --resume-from artifacts/lm_experiments/tinystories_smoke/checkpoints/step_00000500.pt
```

## TinyStories Experiment Batches

Probe which batch sizes fit on a device. This runs short jobs and is intended
to catch obvious OOMs before longer sweeps. The default learning-rate schedule
uses the TinyStories LR sweep winner, `MAX_LR=2e-3` and `MIN_LR=2e-4`.
Runs are written under `artifacts/lm_experiments/probe_batch_size/` by default.
Failed batch sizes are recorded and the probe continues to the next size:

```bash
DEVICE=mps BATCH_SIZES="16 32 64 128 256 512" STEPS=20 scripts/probe_tinystories_batch_sizes.sh
```

Run a learning-rate sweep. `MAX_LRS` controls the tested max learning rates;
the script sets `min_lr = max_lr / 10`. Runs are written under
`artifacts/lm_experiments/grid_search_lr_sweep/` by default.

```bash
DEVICE=mps MAX_LRS="1e-4 2e-4 3e-4 6e-4 1e-3 2e-3 4e-3" scripts/run_tinystories_lr_sweep.sh
```

Run a batch-size sweep at roughly fixed token budget. `TOKENS` defaults to
40.96M, matching the low-resource TinyStories setting. The default learning
rate schedule uses the TinyStories LR sweep winner, `MAX_LR=2e-3` and
`MIN_LR=2e-4`. Runs are written under
`artifacts/lm_experiments/grid_search_batch_size/` by default.

```bash
DEVICE=mps BATCH_SIZES="16 32 64" scripts/run_tinystories_batch_sweep.sh
```

## Experiment Logs

Training scripts use `cs336_basics.experiment.ExperimentLogger` to create local
run directories under `artifacts/lm_experiments/`:

```text
artifacts/lm_experiments/<run_name>/
  config.json
  metrics.jsonl
  samples.jsonl
  checkpoints/
```
