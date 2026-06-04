#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-mps}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"

MAX_ITERS="${MAX_ITERS:-5000}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"

WANDB_MODE="${WANDB_MODE:-offline}"
OUT_DIR="${OUT_DIR:-artifacts/lm_experiments/grid_search_lr_sweep}"
MAX_LRS=(${MAX_LRS:-1e-4 2e-4 3e-4 6e-4 1e-3 2e-3 3e-3})

cd "$ROOT_DIR"

for max_lr in "${MAX_LRS[@]}"; do
  min_lr="$("$PYTHON" -c "v = float('$max_lr') / 10; print(f'{v:.6g}')")"
  lr_name="${max_lr//./p}"
  lr_name="${lr_name//-/m}"
  run_name="tinystories_lr_${lr_name}_${DEVICE}_${RUN_TAG}"

  echo
  echo "== LR sweep max_lr=${max_lr}, min_lr=${min_lr}, device=${DEVICE}, steps=${MAX_ITERS} =="

  "$PYTHON" scripts/train_lm.py \
    --train-data artifacts/lm_data/tinystories_train_10k.npy \
    --valid-data artifacts/lm_data/tinystories_valid_10k.npy \
    --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
    --run-name "$run_name" --out-dir "$OUT_DIR" --device "$DEVICE" --vocab-size 10000 \
    --context-length 256 --batch-size 32 --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 \
    --max-iters "$MAX_ITERS" --warmup-iters "$WARMUP_ITERS" --cosine-cycle-iters "$MAX_ITERS" \
    --max-lr "$max_lr" --min-lr "$min_lr" \
    --log-every 100 --eval-every 1000 --eval-iters 20 --save-every 1000000 \
    --sample-every 0 \
    --wandb-project cs336-a1 --wandb-mode "$WANDB_MODE"
done
