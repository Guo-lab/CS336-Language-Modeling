#!/usr/bin/env bash
set -euo pipefail

BLUE=$'\033[34m'
CYAN=$'\033[36m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

DEVICE="cuda"
CONTEXT_LENGTH="512"
BATCH_SIZE="2"
WARMUP_STEPS="5"
STEPS="10"
SIZES=(small medium large xl 10b)
MODES=(forward forward_backward)
QUICK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --sizes)
      IFS=' ' read -r -a SIZES <<< "$2"
      shift 2
      ;;
    --context-length)
      CONTEXT_LENGTH="$2"
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
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$QUICK" == "1" ]]; then
  echo "${YELLOW}QUICK MODE: tiny CPU settings; not assignment-quality timings.${RESET}"
  DEVICE="cpu"
  CONTEXT_LENGTH="16"
  BATCH_SIZE="2"
  STEPS="2"
  SIZES=(small)
  TINY_ARGS=(--d-model 32 --d-ff 64 --num-layers 1 --num-heads 4 --vocab-size 128)
else
  TINY_ARGS=()
fi

cd "$(dirname "$0")/../.."

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_CMD=(.venv/bin/python)
else
  PYTHON_CMD=(uv run python)
fi

section() {
  echo
  echo "${BLUE}================================================================================${RESET}"
  echo "${BOLD}${CYAN}$1${RESET}"
  echo "${BLUE}================================================================================${RESET}"
}

extract_field() {
  local field="$1"
  awk -F= -v field="$field" '$1 == field { gsub(/[ ,]/, "", $2); print $2 }'
}

run_one() {
  local size="$1"
  local mode="$2"
  local precision="$3"
  local output
  local precision_args=()

  if [[ "$precision" == "bf16" ]]; then
    precision_args=(--mixed-precision --autocast-dtype bfloat16)
  fi

  if ! output=$("${PYTHON_CMD[@]}" scripts/profiling/benchmark_transformer.py \
    --device "$DEVICE" \
    --model-size "$size" \
    --mode "$mode" \
    --context-length "$CONTEXT_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --warmup-steps "$WARMUP_STEPS" \
    --steps "$STEPS" \
    "${precision_args[@]}" \
    "${TINY_ARGS[@]}" 2>&1); then
    if grep -q "OutOfMemoryError\|CUDA out of memory" <<< "$output"; then
      echo "OOM n/a n/a"
    else
      echo "FAILED n/a n/a"
    fi
    return
  fi

  local mean
  local std
  mean=$(printf '%s\n' "$output" | extract_field "mean_ms_per_step")
  std=$(printf '%s\n' "$output" | extract_field "std_ms_per_step")
  echo "OK $mean $std"
}

fmt_value() {
  local value="$1"
  if [[ "$value" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    awk -v value="$value" 'BEGIN { printf "%.3f", value }'
  else
    printf "n/a"
  fi
}

fmt_speedup() {
  local fp32_status="$1"
  local fp32="$2"
  local bf16_status="$3"
  local bf16="$4"
  if [[ "$fp32_status" != "OK" || "$bf16_status" != "OK" ]]; then
    printf "n/a"
    return
  fi
  awk -v fp32="$fp32" -v bf16="$bf16" 'BEGIN { printf "%.2fx", fp32 / bf16 }'
}

section "Mixed precision benchmark: FP32 vs BF16 autocast"
echo "${GREEN}${BOLD}Sanity checks:${RESET}"
"${PYTHON_CMD[@]}" scripts/profiling/mixed_precision_sanity.py
echo

printf "%-8s %-18s %12s %12s %12s %12s %12s\n" \
  "size" "mode" "fp32_ms" "bf16_ms" "speedup" "fp32_std" "bf16_std"
printf "%s\n" "------------------------------------------------------------------------------------------"

for size in "${SIZES[@]}"; do
  for mode in "${MODES[@]}"; do
    read -r fp32_status fp32_mean fp32_std <<< "$(run_one "$size" "$mode" fp32)"
    read -r bf16_status bf16_mean bf16_std <<< "$(run_one "$size" "$mode" bf16)"
    printf "%-8s %-18s %12s %12s %12s %12s %12s\n" \
      "$size" "$mode" \
      "$(fmt_value "$fp32_mean")" \
      "$(fmt_value "$bf16_mean")" \
      "$(fmt_speedup "$fp32_status" "$fp32_mean" "$bf16_status" "$bf16_mean")" \
      "$(fmt_value "$fp32_std")" \
      "$(fmt_value "$bf16_std")"

    if [[ "$fp32_status" != "OK" || "$bf16_status" != "OK" ]]; then
      echo "${YELLOW}${size}/${mode}: fp32=${fp32_status}, bf16=${bf16_status}; unavailable fields shown as n/a.${RESET}"
    fi
  done
done

echo
echo "${GREEN}${BOLD}Answer draft:${RESET}"
echo "BF16 autocast generally reduces matmul/activation memory and can improve runtime, especially for larger matmul-heavy configurations. Configurations that still show n/a are limited by parameter/activation memory on this GPU rather than by timing measurement."
