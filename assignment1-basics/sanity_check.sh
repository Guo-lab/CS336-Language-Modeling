#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

if [[ -x "$ROOT_DIR/.venv-bpe/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv-bpe/bin/python"
fi

BLUE=$'\033[94m'
CYAN=$'\033[96m'
RED=$'\033[91m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

color() {
  if [[ -n "${NO_COLOR:-}" ]]; then
    printf '%s' "$2"
  else
    printf '%s%s%s' "$1" "$2" "$RESET"
  fi
}

sanity_files=()
for path in "$ROOT_DIR"/sanity_check/test_*.py; do
  [[ -e "$path" ]] || continue
  sanity_files+=("$path")
done

print_usage() {
  cat <<EOF
Usage:
  ./sanity_check.sh list
  ./sanity_check.sh all
  ./sanity_check.sh 1 2

Logging:
  CHECK_LOG=1 ./sanity_check.sh all
  CHECK_LOG=1 DESC=bpe_sanity_1 ./sanity_check.sh 1
EOF
}

print_list() {
  color "${BLUE}${BOLD}" "Available sanity checks"
  printf '\n'
  local index=1
  for path in "${sanity_files[@]}"; do
    printf '%2d. %s\n' "$index" "${path#$ROOT_DIR/}"
    index=$((index + 1))
  done
}

run_one() {
  local index="$1"
  local path="${sanity_files[$((index - 1))]}"
  local rel_path="${path#$ROOT_DIR/}"

  printf '\n'
  color "${BLUE}${BOLD}" "== sanity: $rel_path =="
  printf '\n'

  if (cd "$ROOT_DIR" && "$PYTHON" "$rel_path"); then
    color "${CYAN}${BOLD}" "DONE sanity: $rel_path"
    printf '\n'
    return 0
  fi

  color "${RED}${BOLD}" "FAIL sanity: $rel_path"
  printf '\n'
  return 1
}

run_selected() {
  local failed=0
  local selections=("$@")

  color "${BLUE}${BOLD}" "Running sanity checks"
  printf '\npython: %s\n' "$PYTHON"

  for index in "${selections[@]}"; do
    if ! [[ "$index" =~ ^[0-9]+$ ]] || (( index < 1 || index > ${#sanity_files[@]} )); then
      color "${RED}${BOLD}" "Invalid sanity check number: $index"
      printf '\n'
      failed=1
      continue
    fi

    run_one "$index" || failed=1
  done

  return "$failed"
}

make_run_id() {
  local default_suffix="$1"
  local datestamp
  datestamp="$(date +%Y%m%d)"

  if [[ -n "${DESC:-}" ]]; then
    printf '%s_%s_%s\n' "$datestamp" "$DESC" "$default_suffix"
  else
    printf '%s_%s\n' "$datestamp" "$default_suffix"
  fi
}

main() {
  if [[ "$#" -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    return 0
  fi

  if [[ "$1" == "list" ]]; then
    print_list
    return 0
  fi

  local selections=()
  if [[ "$1" == "all" ]]; then
    for ((i = 1; i <= ${#sanity_files[@]}; i++)); do
      selections+=("$i")
    done
  else
    selections=("$@")
  fi

  if [[ "${CHECK_LOG:-0}" == "1" ]]; then
    export NO_COLOR=1
    local log_dir="${LOG_DIR:-$ROOT_DIR/logs}"
    local run_id
    run_id="$(make_run_id sanity)"
    local log_path="$log_dir/${run_id}.log"
    mkdir -p "$log_dir"
    { printf 'log: %s\n' "$log_path"; run_selected "${selections[@]}"; } 2>&1 | tee "$log_path"
    return "${PIPESTATUS[0]}"
  fi

  run_selected "${selections[@]}"
}

main "$@"
