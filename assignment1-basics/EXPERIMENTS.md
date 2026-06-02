# Experiment Runs

Run commands from `assignment1-basics/`.

This file records concrete training/evaluation runs. Keep script usage details in
`scripts/README.md`; keep experiment configs, observations, and follow-up ideas here.

## Environment Checks

MPS check on 2026-05-31:

Stable wheels tested in this environment:
```text
torch 2.9.0   mps built True   mps available False
torch 2.10.0  mps built True   mps available False
torch 2.11.0  mps built True   mps available False
torch 2.12.0  mps built True   mps available False
```

This appears to match the macOS 26 / PyTorch MPS availability issue rather than
a project-code issue.

Nightly install command:

```bash
.venv/bin/python -m pip install --pre --upgrade torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

User terminal check after installing nightly:

```text
torch 2.13.0.dev20260531
mps built True
mps available True
tensor([2., 2.], device='mps:0')
```

W&B is optional. Add `--wandb-project cs336-a1` to upload metrics/samples, or
`--wandb-project cs336-a1 --wandb-mode disabled` to sanity-check the code path
without uploading.

## TinyStories Smoke

Goal: verify the real TinyStories token stream trains end-to-end, with train/valid
loss and text samples logged. This is now closer to the PDF low-resource setting:
`batch_size * max_iters * context_length = 32 * 5000 * 256 = 40.96M` training
tokens.

```bash
.venv/bin/python scripts/train_lm.py \
  --train-data artifacts/lm_data/tinystories_train_10k.npy --valid-data artifacts/lm_data/tinystories_valid_10k.npy --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
  --run-name tinystories_smoke_mps --device mps --vocab-size 10000 \
  --context-length 256 --batch-size 32 --num-layers 4 --d-model 512 --num-heads 8 --d-ff 1344 \
  --max-iters 5000 --warmup-iters 500 --cosine-cycle-iters 5000 --max-lr 3e-4 --min-lr 3e-5 \
  --log-every 50 --eval-every 500 --eval-iters 20 --save-every 1000 \
  --sample-every 500 --sample-prompt "Once upon a time" --sample-max-new-tokens 80 --sample-temperature 0.8 --sample-top-p 0.9
```


### CPU and MPS

```bash
.venv/bin/python scripts/train_lm.py \
  --train-data artifacts/lm_data/tinystories_train_10k.npy --valid-data artifacts/lm_data/tinystories_valid_10k.npy --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
  --run-name tinystories_baseline_h16_mps --device mps --vocab-size 10000 \
  --context-length 256 --batch-size 32 --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 \
  --max-iters 5000 --warmup-iters 500 --cosine-cycle-iters 5000 --max-lr 3e-4 --min-lr 3e-5 \
  --log-every 100 --eval-every 1000 --eval-iters 20 --save-every 2500 \
  --sample-every 2500 --sample-prompt "Once upon a time" --sample-max-new-tokens 80 --sample-temperature 0.8 --sample-top-p 0.9 \
  --wandb-project cs336-a1 --wandb-mode offline
```

```bash
.venv/bin/python scripts/train_lm.py \
  --train-data artifacts/lm_data/tinystories_train_10k.npy --valid-data artifacts/lm_data/tinystories_valid_10k.npy --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
  --run-name tinystories_baseline_h16_cpu --device cpu --vocab-size 10000 \
  --context-length 256 --batch-size 32 --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 \
  --max-iters 5000 --warmup-iters 500 --cosine-cycle-iters 5000 --max-lr 3e-4 --min-lr 3e-5 \
  --log-every 100 --eval-every 1000 --eval-iters 20 --save-every 2500 \
  --sample-every 2500 --sample-prompt "Once upon a time" --sample-max-new-tokens 80 --sample-temperature 0.8 --sample-top-p 0.9 \
  --wandb-project cs336-a1
```

## TinyStories LR Sweep

For the learning-rate sweep, keep the cosine schedule family fixed with
`min_lr = max_lr / 10` and `cosine_cycle_iters = max_iters`.

Observed best run so far:

```text
max_lr=2e-3, min_lr=2e-4, final valid_loss=1.6384
```

This is the default LR schedule for the following TinyStories batch-size sweep.
