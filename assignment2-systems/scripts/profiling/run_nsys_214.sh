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
BATCH_SIZE="2"
WARMUP_STEPS="1"
STEPS="3"
OUT_DIR="artifacts/profiling/nsys_214"
SIZES=(small medium)
CONTEXTS=(256 512 1024)
MODES=(forward)
EXTRA_MODES=(forward_backward full_w_optimizer_step)
EXTRA_SIZE="small"
EXTRA_CONTEXT="512"
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
    --extra-size)
      EXTRA_SIZE="$2"
      shift 2
      ;;
    --extra-context)
      EXTRA_CONTEXT="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    *)
      echo "${RED}unknown arg:${RESET} $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/../.."
mkdir -p "$OUT_DIR"

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

run_profile() {
  local size="$1"
  local context="$2"
  local mode="$3"
  local name="${size}_ctx${context}_${mode}"
  local report="${OUT_DIR}/${name}.nsys-rep"
  local log="${OUT_DIR}/${name}.log"
  local kern="${OUT_DIR}/${name}.cuda_gpu_kern_sum.txt"
  local nvtx="${OUT_DIR}/${name}.nvtx_sum.txt"

  if [[ "$VERBOSE" == "1" ]]; then
    echo "${DIM}$ uv run nsys profile -o ${OUT_DIR}/${name} ...${RESET}" >&2
  fi

  if ! uv run nsys profile \
    --force-overwrite=true \
    -o "${OUT_DIR}/${name}" \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --pytorch=functions-trace,autograd-shapes-nvtx \
    -- "${PYTHON_CMD[@]}" scripts/profiling/benchmark_transformer.py \
      --device "$DEVICE" \
      --model-size "$size" \
      --mode "$mode" \
      --context-length "$context" \
      --batch-size "$BATCH_SIZE" \
      --warmup-steps "$WARMUP_STEPS" \
      --steps "$STEPS" \
      --nvtx \
      --annotate-attention >"$log" 2>&1; then
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$log"; then
      echo "${size} ${context} ${mode} OOM n/a n/a n/a n/a"
    else
      echo "${size} ${context} ${mode} FAILED n/a n/a n/a n/a"
    fi
    return
  fi

  nsys stats --force-export=true --report cuda_gpu_kern_sum "$report" >"$kern" 2>&1 || true
  nsys stats --force-export=true --report nvtx_sum "$report" >"$nvtx" 2>&1 || true

  local mean_ms
  local nvtx_measured_ms
  local top_kernel
  local top_kernel_pct
  local top_kernel_instances
  mean_ms=$(awk -F= '$1 == "mean_ms_per_step" { gsub(/[ ,]/, "", $2); print $2 }' "$log" | tail -n 1)
  nvtx_measured_ms=$(awk '/:benchmark_measured_steps/ { gsub(/,/, "", $2); printf "%.3f", $2 / 1000000.0; exit }' "$nvtx")
  top_kernel_pct=$(awk '/^[[:space:]]+[0-9]/ { print $1; exit }' "$kern")
  top_kernel_instances=$(awk '/^[[:space:]]+[0-9]/ { print $3; exit }' "$kern")
  top_kernel=$(awk '/^[[:space:]]+[0-9]/ {
    name = "";
    for (i = 9; i <= NF; i++) {
      name = name (i == 9 ? "" : " ") $i
    }
    print name;
    exit
  }' "$kern")

  echo "${size} ${context} ${mode} OK ${mean_ms:-n/a} ${nvtx_measured_ms:-n/a} ${top_kernel_pct:-n/a} ${top_kernel_instances:-n/a} ${top_kernel:-n/a}"
}

section "Nsight Systems 2.1.4 runner"
echo "This script intentionally does ${BOLD}not${RESET} use --capture-range=nvtx because NVTX capture did not generate reports on this machine."
echo "It records unfiltered traces and uses NVTX ranges in the report to identify benchmark_measured_steps."
echo "Output directory: ${OUT_DIR}"

section "Forward profiles: two model sizes x three context lengths"
printf "%-8s %8s %-24s %-8s %12s %16s %10s %10s %s\n" \
  "size" "ctx" "mode" "status" "python_ms" "nvtx_ms" "top_%" "top_calls" "top_kernel"
printf "%s\n" "${DIM}------------------------------------------------------------------------------------------------------------------------${RESET}"
for size in "${SIZES[@]}"; do
  for context in "${CONTEXTS[@]}"; do
    read -r r_size r_context r_mode r_status r_mean r_nvtx r_pct r_calls r_kernel <<< "$(run_profile "$size" "$context" forward)"
    printf "%-8s %8s %-24s %-8s %12s %16s %10s %10s %s\n" \
      "$r_size" "$r_context" "$r_mode" "$r_status" "$r_mean" "$r_nvtx" "$r_pct" "$r_calls" "$r_kernel"
  done
done

section "Forward+backward and full-step profiles"
printf "%-8s %8s %-24s %-8s %12s %16s %10s %10s %s\n" \
  "size" "ctx" "mode" "status" "python_ms" "nvtx_ms" "top_%" "top_calls" "top_kernel"
printf "%s\n" "${DIM}------------------------------------------------------------------------------------------------------------------------${RESET}"
for mode in "${EXTRA_MODES[@]}"; do
  read -r r_size r_context r_mode r_status r_mean r_nvtx r_pct r_calls r_kernel <<< "$(run_profile "$EXTRA_SIZE" "$EXTRA_CONTEXT" "$mode")"
  printf "%-8s %8s %-24s %-8s %12s %16s %10s %10s %s\n" \
    "$r_size" "$r_context" "$r_mode" "$r_status" "$r_mean" "$r_nvtx" "$r_pct" "$r_calls" "$r_kernel"
done

section "What to inspect next"
echo "For each successful profile, inspect:"
echo "  ${OUT_DIR}/<name>.cuda_gpu_kern_sum.txt"
echo "  ${OUT_DIR}/<name>.nvtx_sum.txt"
echo
echo "Useful greps:"
echo "  grep 'benchmark_measured_steps\\|scaled dot product attention\\|computing softmax\\|final matmul' ${OUT_DIR}/*.nvtx_sum.txt"
echo "  head -n 30 ${OUT_DIR}/small_ctx512_forward.cuda_gpu_kern_sum.txt"
