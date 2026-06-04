#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
TOKENS="${TOKENS:-40960000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
WANDB_MODE="${WANDB_MODE:-offline}"
OUT_DIR="${OUT_DIR:-artifacts/lm_experiments/grid_search_batch_size}"
BATCH_SIZES=(${BATCH_SIZES:-16 32 64})
MAX_LR="${MAX_LR:-2e-3}"
MIN_LR="${MIN_LR:-2e-4}"

cd "$ROOT_DIR"

for batch_size in "${BATCH_SIZES[@]}"; do
  max_iters="$("$PYTHON" -c "print(max(1, round($TOKENS / ($batch_size * $CONTEXT_LENGTH))))")"
  warmup_iters="$("$PYTHON" -c "print(max(1, round($max_iters * 0.1)))")"
  run_name="tinystories_bs${batch_size}_${DEVICE}_${RUN_TAG}"

  echo
  echo "== Batch sweep batch_size=${batch_size}, steps=${max_iters}, device=${DEVICE}, max_lr=${MAX_LR}, min_lr=${MIN_LR} =="

  "$PYTHON" scripts/train_lm.py \
    --train-data artifacts/lm_data/tinystories_train_10k.npy --valid-data artifacts/lm_data/tinystories_valid_10k.npy --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
    --run-name "$run_name" --out-dir "$OUT_DIR" --device "$DEVICE" --vocab-size 10000 \
    --context-length "$CONTEXT_LENGTH" --batch-size "$batch_size" --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 \
    --max-iters "$max_iters" --warmup-iters "$warmup_iters" --cosine-cycle-iters "$max_iters" --max-lr "$MAX_LR" --min-lr "$MIN_LR" \
    --log-every 100 --eval-every 1000 --eval-iters 20 --save-every 1000000 \
    --sample-every 0 \
    --wandb-project cs336-a1 --wandb-mode "$WANDB_MODE"
done
