#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${CS336_SOURCE_DIR:-$REPO_ROOT/source-cs336}"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory does not exist: $SOURCE_DIR" >&2
  exit 1
fi

while IFS= read -r git_dir; do
  repo_dir="$(dirname "$git_dir")"
  echo "Pulling ${repo_dir#$SOURCE_DIR/}"
  git -C "$repo_dir" pull --ff-only
done < <(find "$SOURCE_DIR" -mindepth 2 -maxdepth 2 -type d -name .git | sort)

rsync -a --ignore-existing \
  --exclude='.git/' \
  --exclude='.DS_Store' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.venv/' \
  --exclude='node_modules/' \
  "$SOURCE_DIR"/ "$REPO_ROOT"/

source_files() {
  find "$SOURCE_DIR" \
    \( -name .git -o -name __pycache__ -o -name .pytest_cache -o -name .venv -o -name node_modules \) -prune \
    -o -type f \
    ! -name .DS_Store \
    ! -name '*.pyc' \
    -print0
}

repo_files() {
  find "$REPO_ROOT" \
    \( -path "$REPO_ROOT/.git" -o -path "$SOURCE_DIR" -o -path "$REPO_ROOT/scripts" \
       -o -name __pycache__ -o -name .pytest_cache -o -name .venv -o -name node_modules \) -prune \
    -o -type f \
    ! -name .DS_Store \
    ! -name '*.pyc' \
    ! -path "$REPO_ROOT/.gitignore" \
    -print0
}

print_header() {
  echo
  echo "$1"
}

print_header "Existing files that differ from source:"
found=0
while IFS= read -r -d '' source_file; do
  rel="${source_file#$SOURCE_DIR/}"
  target_file="$REPO_ROOT/$rel"

  if [[ -f "$target_file" ]] && ! cmp -s "$source_file" "$target_file"; then
    echo "  $rel"
    found=1
  fi
done < <(source_files)
[[ "$found" -eq 0 ]] && echo "  none"

print_header "Files that exist only in this repo:"
found=0
while IFS= read -r -d '' repo_file; do
  rel="${repo_file#$REPO_ROOT/}"

  if [[ ! -e "$SOURCE_DIR/$rel" ]]; then
    echo "  $rel"
    found=1
  fi
done < <(repo_files)
[[ "$found" -eq 0 ]] && echo "  none"
