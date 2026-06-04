#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
STEPS="${STEPS:-20}"
WANDB_MODE="${WANDB_MODE:-offline}"
OUT_DIR="${OUT_DIR:-artifacts/lm_experiments/probe_batch_size}"
BATCH_SIZES=(${BATCH_SIZES:-16 32 64 128 256 512})
MAX_LR="${MAX_LR:-2e-3}"
MIN_LR="${MIN_LR:-2e-4}"

cd "$ROOT_DIR"

summary_file="$OUT_DIR/probe_${DEVICE}_${RUN_TAG}.tsv"
mkdir -p "$OUT_DIR"
printf "batch_size\tstatus\n" > "$summary_file"

for batch_size in "${BATCH_SIZES[@]}"; do
  run_name="probe_tinystories_h16_${DEVICE}_bs${batch_size}_${RUN_TAG}"
  echo
  echo "== Probe batch_size=${batch_size}, device=${DEVICE}, steps=${STEPS}, max_lr=${MAX_LR}, min_lr=${MIN_LR} =="

  if "$PYTHON" scripts/train_lm.py \
    --train-data artifacts/lm_data/tinystories_train_10k.npy --valid-data artifacts/lm_data/tinystories_valid_10k.npy --tokenizer artifacts/tokenizer_experiments/training/tokenizers_chunked_mp8/tinystories_train_10k \
    --run-name "$run_name" --out-dir "$OUT_DIR" --device "$DEVICE" --vocab-size 10000 \
    --context-length 256 --batch-size "$batch_size" --num-layers 4 --d-model 512 --num-heads 16 --d-ff 1344 \
    --max-iters "$STEPS" --warmup-iters 2 --cosine-cycle-iters "$STEPS" --max-lr "$MAX_LR" --min-lr "$MIN_LR" \
    --log-every "$STEPS" --eval-every 1000000 --eval-iters 1 --save-every 1000000 \
    --sample-every 0 \
    --wandb-project cs336-a1 --wandb-mode "$WANDB_MODE"; then
    printf "%s\tPASS\n" "$batch_size" | tee -a "$summary_file"
  else
    printf "%s\tFAIL\n" "$batch_size" | tee -a "$summary_file"
  fi
done

echo
echo "Probe summary: $summary_file"
