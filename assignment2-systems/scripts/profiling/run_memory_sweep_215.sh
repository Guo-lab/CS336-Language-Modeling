#!/usr/bin/env bash
set -euo pipefail

BLUE=$'\033[34m'
CYAN=$'\033[36m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

DEVICE="cuda"
BATCH_SIZE="1"
WARMUP_STEPS="1"
STEPS="1"
OUT_DIR="artifacts/profiling/memory_215"
LOG_DIR="${OUT_DIR}/logs"
SIZES=(small medium large xl)
CONTEXTS=(128 2048)
MODES=(forward full_w_optimizer_step)
RUN_BFLOAT16_STORAGE=1
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --warmup-steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      LOG_DIR="${OUT_DIR}/logs"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --sizes)
      IFS=' ' read -r -a SIZES <<< "$2"
      shift 2
      ;;
    --contexts)
      IFS=' ' read -r -a CONTEXTS <<< "$2"
      shift 2
      ;;
    --modes)
      IFS=' ' read -r -a MODES <<< "$2"
      shift 2
      ;;
    --no-bfloat16-storage)
      RUN_BFLOAT16_STORAGE=0
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/../.."
mkdir -p "$LOG_DIR"

section() {
  echo
  echo "${BLUE}================================================================================${RESET}"
  echo "${BOLD}${CYAN}$1${RESET}"
  echo "${BLUE}================================================================================${RESET}"
}

run_for_size() {
  local size="$1"
  local log="${LOG_DIR}/${size}.txt"
  local contexts_string="${CONTEXTS[*]}"
  local modes_string="${MODES[*]}"

  {
    echo "Memory profiling sweep for model_size=${size}"
    echo "device=${DEVICE}"
    echo "batch_size=${BATCH_SIZE}"
    echo "warmup_steps=${WARMUP_STEPS}"
    echo "steps=${STEPS}"
    echo "contexts=${contexts_string}"
    echo "modes=${modes_string}"
    echo
    echo "===== float32 parameter storage: fp32 vs bf16 autocast ====="
    NO_COLOR=1 scripts/profiling/run_memory_profiling_215.sh \
      --no-color \
      --device "$DEVICE" \
      --model-size "$size" \
      --dtype float32 \
      --contexts "$contexts_string" \
      --modes "$modes_string" \
      --precisions "fp32 bf16" \
      --batch-size "$BATCH_SIZE" \
      --warmup-steps "$WARMUP_STEPS" \
      --steps "$STEPS" \
      --out-dir "$OUT_DIR"

    if [[ "$RUN_BFLOAT16_STORAGE" == "1" ]]; then
      echo
      echo "===== bfloat16 parameter storage: bf16 autocast ====="
      NO_COLOR=1 scripts/profiling/run_memory_profiling_215.sh \
        --no-color \
        --device "$DEVICE" \
        --model-size "$size" \
        --dtype bfloat16 \
        --contexts "$contexts_string" \
        --modes "$modes_string" \
        --precisions "bf16" \
        --batch-size "$BATCH_SIZE" \
        --warmup-steps "$WARMUP_STEPS" \
        --steps "$STEPS" \
        --out-dir "$OUT_DIR"
    fi
  } >"$log" 2>&1

  echo "$log"
}

section "Memory profiling sweep"
echo "Each model size writes one plain-text log under ${LOG_DIR}."
echo "sizes=${SIZES[*]}"
echo "contexts=${CONTEXTS[*]}"
echo "modes=${MODES[*]}"

for size in "${SIZES[@]}"; do
  if [[ "$VERBOSE" == "1" ]]; then
    echo "${YELLOW}Running size=${size}...${RESET}"
  fi
  log=$(run_for_size "$size")
  echo "${GREEN}${size}:${RESET} ${log}"
done
