#!/usr/bin/env bash
set -euo pipefail

BLUE=$'\033[34m'
CYAN=$'\033[36m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'
BOLD=$'\033[1m'
DIM=$'\033[2m'
RESET=$'\033[0m'

DEVICE="cuda"
CONTEXT_LENGTH="512"
BATCH_SIZE="2" # Even after reducing the batch size from the default 4 to 1 or 2, the XL and 10B models still OOM on 4060.
WARMUP_STEPS="5"
STEPS="10"
C_SIZE="small"
C_MODE="full_w_optimizer_step"
QUICK=0
VERBOSE=0
SIZES=(small medium large xl 10b)
MODES=(forward forward_backward full_w_optimizer_step)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --verbose)
      VERBOSE=1
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
    --c-size)
      C_SIZE="$2"
      shift 2
      ;;
    --c-mode)
      C_MODE="$2"
      shift 2
      ;;
    *)
      echo "${RED}unknown arg:${RESET} $1" >&2
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
  local warmup="$3"
  local output

  if [[ "$VERBOSE" == "1" ]]; then
    echo "${DIM}$ ${PYTHON_CMD[*]} scripts/profiling/benchmark_transformer.py --device $DEVICE --model-size $size --mode $mode --context-length $CONTEXT_LENGTH --batch-size $BATCH_SIZE --warmup-steps $warmup --steps $STEPS ${TINY_ARGS[*]}${RESET}" >&2
  fi

  if ! output=$("${PYTHON_CMD[@]}" scripts/profiling/benchmark_transformer.py \
    --device "$DEVICE" \
    --model-size "$size" \
    --mode "$mode" \
    --context-length "$CONTEXT_LENGTH" \
    --batch-size "$BATCH_SIZE" \
    --warmup-steps "$warmup" \
    --steps "$STEPS" \
    "${TINY_ARGS[@]}" 2>&1); then
    if grep -q "OutOfMemoryError\|CUDA out of memory" <<< "$output"; then
      echo "OOM nan nan nan"
    else
      echo "FAILED nan nan nan"
    fi
    return
  fi

  local mean
  local std
  local tokens
  mean=$(printf '%s\n' "$output" | extract_field "mean_ms_per_step")
  std=$(printf '%s\n' "$output" | extract_field "std_ms_per_step")
  tokens=$(printf '%s\n' "$output" | extract_field "tokens_per_second")
  echo "OK $mean $std $tokens"
}

is_number() {
  [[ "$1" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
}

fmt_ms() {
  local value="$1"
  if ! is_number "$value"; then
    printf "n/a"
    return
  fi
  awk -v value="$value" 'BEGIN { printf "%.3f", value }'
}

fmt_diff_ms() {
  local lhs_status="$1"
  local lhs="$2"
  local rhs_status="$3"
  local rhs="$4"
  if [[ "$lhs_status" != "OK" || "$rhs_status" != "OK" ]]; then
    printf "n/a"
    return
  fi
  awk -v lhs="$lhs" -v rhs="$rhs" 'BEGIN { printf "%.3f", lhs - rhs }'
}

fmt_ratio() {
  local mean="$1"
  local std="$2"
  if ! is_number "$mean" || ! is_number "$std"; then
    printf "n/a"
    return
  fi
  awk -v mean="$mean" -v std="$std" '
    BEGIN {
      if (mean <= 0) {
        printf "n/a"
      } else {
        ratio = std / mean * 100
        printf "%.3f (%4.1f%%)", std, ratio
      }
    }'
}

section "(a) Script capability check"
printf "%-18s : %s\n" "${BOLD}model init${RESET}" "via --model-size or explicit hyperparameter overrides"
printf "%-18s : %s\n" "${BOLD}random batch${RESET}" "make_random_batch creates token_ids and targets"
printf "%-18s : %s\n" "${BOLD}warmup/timing${RESET}" "--warmup-steps and --steps"
printf "%-18s : %s\n" "${BOLD}modes${RESET}" "${MODES[*]}"
printf "%-18s : %s\n" "${BOLD}cuda sync${RESET}" "after warmup and after every measured step"

section "(b) Timings: 5 warmups, 10 measured steps"
printf "%-8s %12s %12s %12s %18s %18s %18s\n" "size" "forward" "backward" "opt_step" "forward_std" "fwd+bwd_std" "full_std"
printf "%s\n" "${DIM}--------------------------------------------------------------------------------------------------------${RESET}"

for size in "${SIZES[@]}"; do
  read -r fwd_status fwd_mean fwd_std _ <<< "$(run_one "$size" forward "$WARMUP_STEPS")"
  read -r fb_status fb_mean fb_std _ <<< "$(run_one "$size" forward_backward "$WARMUP_STEPS")"
  read -r full_status full_mean full_std _ <<< "$(run_one "$size" full_w_optimizer_step "$WARMUP_STEPS")"

  bwd_mean=$(fmt_diff_ms "$fb_status" "$fb_mean" "$fwd_status" "$fwd_mean")
  opt_mean=$(fmt_diff_ms "$full_status" "$full_mean" "$fb_status" "$fb_mean")
  printf "%-8s %12s %12s %12s %18s %18s %18s\n" \
    "$size" "$(fmt_ms "$fwd_mean")" "$(fmt_ms "$bwd_mean")" "$(fmt_ms "$opt_mean")" \
    "$(fmt_ratio "$fwd_mean" "$fwd_std")" \
    "$(fmt_ratio "$fb_mean" "$fb_std")" \
    "$(fmt_ratio "$full_mean" "$full_std")"

  if [[ "$fwd_status" != "OK" || "$fb_status" != "OK" || "$full_status" != "OK" ]]; then
    echo "${YELLOW}${size}: ${fwd_status}/forward, ${fb_status}/forward_backward, ${full_status}/full_w_optimizer_step -> unavailable fields shown as n/a.${RESET}"
  fi
done

echo

section "(c) Warmup ablation"
printf "%8s %12s %18s %14s\n" "warmup" "mean_ms" "std" "tokens/s"
printf "%s\n" "${DIM}--------------------------------------------------------${RESET}"

for warmup in 0 1 2 5; do
  read -r status mean std tokens <<< "$(run_one "$C_SIZE" "$C_MODE" "$warmup")"
  if [[ "$status" != "OK" ]]; then
    printf "%8s %12s %18s %14s\n" "$warmup" "n/a" "n/a" "n/a"
    echo "${YELLOW}${C_SIZE}/${C_MODE}/warmup=${warmup}: ${status} -> shown as n/a.${RESET}"
    continue
  fi
  printf "%8s %12.3f %18s %14.2f\n" "$warmup" "$mean" "$(fmt_ratio "$mean" "$std")" "$tokens"
done

echo

