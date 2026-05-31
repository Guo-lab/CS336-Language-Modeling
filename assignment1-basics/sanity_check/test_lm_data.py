"""Manual sanity check for tokenized LM `.npy` datasets.

Run from the assignment root:
    python sanity_check/test_lm_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.tokenizer import Tokenizer  # noqa: E402
from sanity_check.utils import fail_text, pass_text, title, warn_text  # noqa: E402

DEFAULT_PAIRS = [
    (
        ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        ROOT / "artifacts" / "lm_data" / "tinystories_train_10k.npy",
        ROOT
        / "artifacts"
        / "tokenizer_experiments"
        / "training"
        / "tokenizers_chunked_mp8"
        / "tinystories_train_10k",
    ),
    (
        ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt",
        ROOT / "artifacts" / "lm_data" / "tinystories_valid_10k.npy",
        ROOT
        / "artifacts"
        / "tokenizer_experiments"
        / "training"
        / "tokenizers_chunked_mp8"
        / "tinystories_train_10k",
    ),
    (
        ROOT / "data" / "owt_train.txt",
        ROOT / "artifacts" / "lm_data" / "owt_train_32k.npy",
        ROOT
        / "artifacts"
        / "tokenizer_experiments"
        / "training"
        / "tokenizers_chunked_mp8"
        / "owt_train_32k",
    ),
    (
        ROOT / "data" / "owt_valid.txt",
        ROOT / "artifacts" / "lm_data" / "owt_valid_32k.npy",
        ROOT
        / "artifacts"
        / "tokenizer_experiments"
        / "training"
        / "tokenizers_chunked_mp8"
        / "owt_train_32k",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect tokenized LM data and decode a preview."
    )
    parser.add_argument("--preview-tokens", type=int, default=80)
    parser.add_argument("--source-chars", type=int, default=20_000)
    return parser


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_files(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merges.json"),
        special_tokens=["<|endoftext|>"],
    )


def read_source_prefix(path: Path, max_chars: int) -> str:
    with path.open(encoding="utf-8") as f:
        return f.read(max_chars)


def expected_token_prefix(
    source_path: Path,
    tokenizer: Tokenizer,
    preview_tokens: int,
    source_chars: int,
) -> tuple[list[int], str]:
    source_text = read_source_prefix(source_path, source_chars)
    encoded = tokenizer.encode(source_text)
    if len(encoded) < preview_tokens:
        raise ValueError(
            f"Source prefix from {source_path} produced only {len(encoded)} tokens; "
            f"increase --source-chars"
        )
    expected_ids = encoded[:preview_tokens]
    expected_text = tokenizer.decode(expected_ids)
    return expected_ids, expected_text


def check_pair(
    source_path: Path,
    data_path: Path,
    tokenizer_dir: Path,
    preview_tokens: int,
    source_chars: int,
) -> bool:
    print(title(f"\n== {data_path.name} =="))

    if not source_path.exists():
        print(warn_text(f"SKIP missing source text: {source_path}"))
        return False
    if not data_path.exists():
        print(warn_text(f"SKIP missing data: {data_path}"))
        return False
    if not tokenizer_dir.exists():
        print(warn_text(f"SKIP missing tokenizer: {tokenizer_dir}"))
        return False

    tokens = np.load(data_path, mmap_mode="r")
    tokenizer = load_tokenizer(tokenizer_dir)
    actual_ids = tokens[:preview_tokens].astype(int).tolist()
    actual_text = tokenizer.decode(actual_ids)
    expected_ids, expected_text = expected_token_prefix(
        source_path,
        tokenizer,
        preview_tokens,
        source_chars,
    )
    ids_match = actual_ids == expected_ids
    text_matches = actual_text == expected_text
    mismatch_idx = None
    if not ids_match:
        for idx, (expected, actual) in enumerate(zip(expected_ids, actual_ids)):
            if expected != actual:
                mismatch_idx = idx
                break

    print(f"source path:   {source_path.relative_to(ROOT)}")
    print(f"data path:     {data_path.relative_to(ROOT)}")
    print(f"tokenizer:     {tokenizer_dir.relative_to(ROOT)}")
    print(f"shape:         {tokens.shape}")
    print(f"dtype:         {tokens.dtype}")
    print(f"expected ids:  {expected_ids[:20]}")
    print(f"actual ids:    {actual_ids[:20]}")
    if mismatch_idx is not None:
        start = max(0, mismatch_idx - 5)
        end = min(preview_tokens, mismatch_idx + 6)
        print(f"first mismatch token offset: {mismatch_idx}")
        print(f"expected ids around mismatch: {expected_ids[start:end]}")
        print(f"actual ids around mismatch:   {actual_ids[start:end]}")
    print("expected decoded text:")
    print(expected_text[:500])
    print("actual decoded text:")
    print(actual_text[:500])
    if ids_match:
        print(pass_text("PASS exact token-id match"))
        return True
    if text_matches:
        print(warn_text("SOFT PASS token ids differ, decoded text matches"))
        return True

    print(fail_text("FAIL decoded text differs"))
    return False


def main() -> int:
    args = build_parser().parse_args()
    checked = 0

    print(title("LM data sanity check"))
    print("Compares tokenized .npy prefixes against freshly encoded source text.\n")

    for source_path, data_path, tokenizer_dir in DEFAULT_PAIRS:
        checked += check_pair(
            source_path,
            data_path,
            tokenizer_dir,
            args.preview_tokens,
            args.source_chars,
        )

    if checked == 0:
        print(warn_text("\nNo tokenized datasets were available to check."))
        return 1

    print(pass_text(f"\nChecked {checked} dataset(s)."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
