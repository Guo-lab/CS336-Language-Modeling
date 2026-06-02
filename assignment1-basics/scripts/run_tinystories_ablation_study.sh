#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
WANDB_MODE="${WANDB_MODE:-offline}"
OUT_DIR="${OUT_DIR:-artifacts/lm_experiments/ablation_study}"

MAX_ITERS="${MAX_ITERS:-5000}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"
MAX_LR="${MAX_LR:-2e-3}"
MIN_LR="${MIN_LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"

cd "$ROOT_DIR"

run_ablation() {
  local name="$1"
  shift

  local run_name="tinystories_ablate_${name}_${DEVICE}_${RUN_TAG}"
  echo
  echo "== Ablation ${name}: device=${DEVICE}, steps=${MAX_ITERS}, max_lr=${MAX_LR}, min_lr=${MIN_LR} =="

  "$PYTHON" scripts/train_lm.py \
    --train-data artifacts/lm_data/tinystories_train_10k.npy \
    --valid-data artifacts/lm_data/tinystories_valid_10k.npy \
    --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
    --run-name "$run_name" --out-dir "$OUT_DIR" --device "$DEVICE" --vocab-size 10000 \
    --context-length "$CONTEXT_LENGTH" --batch-size "$BATCH_SIZE" \
    --num-layers 4 --d-model 512 --num-heads 16 \
    --max-iters "$MAX_ITERS" --warmup-iters "$WARMUP_ITERS" --cosine-cycle-iters "$MAX_ITERS" \
    --max-lr "$MAX_LR" --min-lr "$MIN_LR" \
    --log-every 100 --eval-every 1000 --eval-iters 20 --save-every 1000000 \
    --sample-every 0 \
    --wandb-project cs336-a1 --wandb-mode "$WANDB_MODE" \
    "$@"
}

run_ablation "no_rmsnorm" --d-ff 1344 --norm-mode none --position-encoding rope --ffn-type swiglu
run_ablation "post_norm" --d-ff 1344 --norm-mode post --position-encoding rope --ffn-type swiglu
run_ablation "nope" --d-ff 1344 --norm-mode pre --position-encoding none --ffn-type swiglu
run_ablation "silu_ffn" --d-ff 2048 --norm-mode pre --position-encoding rope --ffn-type silu
