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
MODEL_SIZE="xl"
MODEL_DTYPE="float32"
BATCH_SIZE="1"
WARMUP_STEPS="1"
STEPS="1"
OUT_DIR="artifacts/profiling/memory_215"
CONTEXTS=(128 2048)
MODES=(forward full_w_optimizer_step)
PRECISIONS=(fp32 bf16)
RUN_NSYS=0
VERBOSE=0
NO_COLOR_FLAG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --model-size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --dtype)
      MODEL_DTYPE="$2"
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
    --contexts)
      IFS=' ' read -r -a CONTEXTS <<< "$2"
      shift 2
      ;;
    --modes)
      IFS=' ' read -r -a MODES <<< "$2"
      shift 2
      ;;
    --precisions)
      IFS=' ' read -r -a PRECISIONS <<< "$2"
      shift 2
      ;;
    --run-nsys)
      RUN_NSYS=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    --no-color)
      NO_COLOR_FLAG=1
      shift
      ;;
    *)
      echo "${RED}unknown arg:${RESET} $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$NO_COLOR_FLAG" == "1" || -n "${NO_COLOR:-}" || ! -t 1 ]]; then
  BLUE=""
  CYAN=""
  GREEN=""
  YELLOW=""
  RED=""
  BOLD=""
  DIM=""
  RESET=""
fi

cd "$(dirname "$0")/../.."
mkdir -p "$OUT_DIR"

if [[ -x ".venv/bin/python3" ]]; then
  PYTHON_CMD=(.venv/bin/python3)
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

fmt_value() {
  local value="$1"
  if [[ "$value" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    awk -v value="$value" 'BEGIN { printf "%.3f", value }'
  else
    printf "n/a"
  fi
}

run_snapshot() {
  local context="$1"
  local mode="$2"
  local precision="$3"
  local name="${MODEL_SIZE}_${MODEL_DTYPE}_ctx${context}_${mode}_${precision}"
  local snapshot="${OUT_DIR}/${name}.pickle"
  local log="${OUT_DIR}/${name}.log"
  local precision_opts=()
  local output

  if [[ "$precision" == "bf16" ]]; then
    precision_opts=(--mixed-precision --autocast-dtype bfloat16)
  fi

  if [[ "$VERBOSE" == "1" ]]; then
    echo "${DIM}$ ${PYTHON_CMD[*]} scripts/profiling/benchmark_transformer.py --model-size ${MODEL_SIZE} --context-length ${context} --mode ${mode} --memory-profile ...${RESET}" >&2
  fi

  if ! output=$("${PYTHON_CMD[@]}" scripts/profiling/benchmark_transformer.py \
    --device "$DEVICE" \
    --model-size "$MODEL_SIZE" \
    --dtype "$MODEL_DTYPE" \
    --mode "$mode" \
    --context-length "$context" \
    --batch-size "$BATCH_SIZE" \
    --warmup-steps "$WARMUP_STEPS" \
    --steps "$STEPS" \
    --memory-profile \
    --memory-snapshot-path "$snapshot" \
    "${precision_opts[@]}" 2>&1); then
    printf "%s\n" "$output" >"$log"
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$log"; then
      echo "${context} ${mode} ${precision} OOM n/a n/a n/a ${log}"
    else
      echo "${context} ${mode} ${precision} FAILED n/a n/a n/a ${log}"
    fi
    return
  fi

  printf "%s\n" "$output" >"$log"

  local peak_allocated
  local peak_reserved
  local mean_ms
  peak_allocated=$(printf "%s\n" "$output" | extract_field "peak_memory_allocated_gib" | tail -n 1)
  peak_reserved=$(printf "%s\n" "$output" | extract_field "peak_memory_reserved_gib" | tail -n 1)
  mean_ms=$(printf "%s\n" "$output" | extract_field "mean_ms_per_step" | tail -n 1)

  echo "${context} ${mode} ${precision} OK ${peak_allocated:-n/a} ${peak_reserved:-n/a} ${mean_ms:-n/a} ${snapshot}"
}

run_nsys_memory_profile() {
  local context="$1"
  local mode="$2"
  local name="${MODEL_SIZE}_${MODEL_DTYPE}_ctx${context}_${mode}_nsys_memory"
  local log="${OUT_DIR}/${name}.log"

  uv run nsys profile \
    --force-overwrite=true \
    -o "${OUT_DIR}/${name}" \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --pytorch=functions-trace,autograd-shapes-nvtx \
    --cuda-memory-usage=true \
    -- "${PYTHON_CMD[@]}" scripts/profiling/benchmark_transformer.py \
      --device "$DEVICE" \
      --model-size "$MODEL_SIZE" \
      --dtype "$MODEL_DTYPE" \
      --mode "$mode" \
      --context-length "$context" \
      --batch-size "$BATCH_SIZE" \
      --warmup-steps "$WARMUP_STEPS" \
      --steps "$STEPS" \
      --nvtx >"$log" 2>&1 || true
}

section "PyTorch memory snapshots for 2.1.5"
echo "Output directory: ${OUT_DIR}"
echo "Each successful .pickle can be opened at pytorch.org/memory_viz."
echo "Defaults are conservative: model=${MODEL_SIZE}, dtype=${MODEL_DTYPE}, batch_size=${BATCH_SIZE}, warmup=${WARMUP_STEPS}, steps=${STEPS}."

section "Peak memory table"
printf "%-8s %-24s %-8s %-8s %14s %14s %12s %s\n" \
  "ctx" "mode" "precision" "status" "peak_alloc" "peak_reserved" "mean_ms" "artifact"
printf "%s\n" "${DIM}------------------------------------------------------------------------------------------------------------------------${RESET}"

for context in "${CONTEXTS[@]}"; do
  for mode in "${MODES[@]}"; do
    for precision in "${PRECISIONS[@]}"; do
      read -r r_context r_mode r_precision r_status r_peak r_reserved r_mean r_artifact <<< "$(run_snapshot "$context" "$mode" "$precision")"
      printf "%-8s %-24s %-8s %-8s %14s %14s %12s %s\n" \
        "$r_context" "$r_mode" "$r_precision" "$r_status" \
        "$(fmt_value "$r_peak")" "$(fmt_value "$r_reserved")" "$(fmt_value "$r_mean")" "$r_artifact"
      if [[ "$r_status" != "OK" ]]; then
        echo "${YELLOW}${MODEL_SIZE}/ctx${r_context}/${r_mode}/${r_precision}: ${r_status}; traceback saved to ${r_artifact}.${RESET}"
      fi
    done
  done
done

section "XL reference residual stream activation size"
"${PYTHON_CMD[@]}" - "$BATCH_SIZE" <<'PY'
import sys

batch_size = int(sys.argv[1])
d_model = 2560
bytes_per_fp32 = 4
for context in (128, 2048):
    mib = batch_size * context * d_model * bytes_per_fp32 / 1024**2
    print(
        f"batch={batch_size}, context={context}: "
        f"{batch_size} x {context} x {d_model} x 4 bytes = {mib:.3f} MiB"
    )
PY

section "Nsight memory profiling for part (f)"
if [[ "$RUN_NSYS" == "1" ]]; then
  echo "Running one Nsight memory trace for ${MODEL_SIZE}, ctx=128, full_w_optimizer_step."
  run_nsys_memory_profile "128" "full_w_optimizer_step"
  echo "${GREEN}Nsight output prefix:${RESET} ${OUT_DIR}/${MODEL_SIZE}_${MODEL_DTYPE}_ctx128_full_w_optimizer_step_nsys_memory"
else
  echo "Not running Nsight by default because reports can be large."
  echo "Run this script with ${BOLD}--run-nsys${RESET}, or run the command template below:"
  echo
  cat <<EOF
uv run nsys profile \\
  --force-overwrite=true \\
  -o ${OUT_DIR}/${MODEL_SIZE}_${MODEL_DTYPE}_ctx128_full_w_optimizer_step_nsys_memory \\
  --trace=cuda,cudnn,cublas,osrt,nvtx \\
  --pytorch=functions-trace,autograd-shapes-nvtx \\
  --cuda-memory-usage=true \\
  -- python scripts/profiling/benchmark_transformer.py \\
    --device ${DEVICE} \\
    --model-size ${MODEL_SIZE} \\
    --dtype ${MODEL_DTYPE} \\
    --mode full_w_optimizer_step \\
    --context-length 128 \\
    --batch-size ${BATCH_SIZE} \\
    --warmup-steps ${WARMUP_STEPS} \\
    --steps ${STEPS} \\
    --nvtx
EOF
fi
