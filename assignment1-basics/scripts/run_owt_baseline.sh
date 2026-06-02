#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
WANDB_MODE="${WANDB_MODE:-offline}"
OUT_DIR="${OUT_DIR:-artifacts/lm_experiments/owt_baseline}"

MAX_ITERS="${MAX_ITERS:-5000}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"
MAX_LR="${MAX_LR:-2e-3}"
MIN_LR="${MIN_LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"

cd "$ROOT_DIR"

run_name="owt_baseline_${DEVICE}_${RUN_TAG}"

echo
echo "== OWT baseline: device=${DEVICE}, steps=${MAX_ITERS}, batch_size=${BATCH_SIZE}, max_lr=${MAX_LR}, min_lr=${MIN_LR} =="

"$PYTHON" scripts/train_lm.py \
  --train-data artifacts/lm_data/owt_train_32k.npy \
  --valid-data artifacts/lm_data/owt_valid_32k.npy \
  --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/owt_train_32k \
  --run-name "$run_name" --out-dir "$OUT_DIR" --device "$DEVICE" --vocab-size 32000 \
  --context-length "$CONTEXT_LENGTH" --batch-size "$BATCH_SIZE" \
  --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 \
  --max-iters "$MAX_ITERS" --warmup-iters "$WARMUP_ITERS" --cosine-cycle-iters "$MAX_ITERS" \
  --max-lr "$MAX_LR" --min-lr "$MIN_LR" \
  --log-every 100 --eval-every 1000 --eval-iters 20 --save-every 2500 \
  --sample-every 1000 --sample-prompt "The" --sample-max-new-tokens 120 --sample-temperature 0.8 --sample-top-p 0.9 \
  --wandb-project cs336-a1 --wandb-mode "$WANDB_MODE"
