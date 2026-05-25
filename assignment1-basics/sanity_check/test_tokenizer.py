"""Manual sanity checks for cs336_basics.tokenizer.

Run from the assignment root:
    python sanity_check/test_tokenizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.tokenizer import Tokenizer  # noqa: E402
from sanity_check.utils import fail_text, pass_text, title, warn_text  # noqa: E402


OWT_TOKENIZER_DIR = ROOT / "artifacts" / "tokenizers_chunked_mp8" / "owt_train_32k"


def print_check(name: str, expected, result) -> bool:
    print(title(f"\n== {name} =="))
    print(f"expected: {expected!r}")
    print(f"result:   {result!r}")
    ok = result == expected
    print(pass_text() if ok else fail_text())
    return ok


def check_init_stores_fields() -> bool:
    vocab = {0: b"a", 1: b"b", 2: b"ab"}
    merges = [(b"a", b"b")]
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    checks = [
        print_check("init stores vocab", vocab, tokenizer.vocab),
        print_check("init stores merges", merges, tokenizer.merges),
        print_check("init stores special tokens", special_tokens, tokenizer.special_tokens),
    ]
    return all(checks)


def check_from_files_loads_artifacts() -> bool:
    tokenizer = Tokenizer.from_files(
        str(OWT_TOKENIZER_DIR / "vocab.json"),
        str(OWT_TOKENIZER_DIR / "merges.json"),
        ["<|endoftext|>"],
    )

    checks = [
        print_check("from_files vocab size", 32000, len(tokenizer.vocab)),
        print_check("from_files merge count", 31743, len(tokenizer.merges)),
        print_check("first byte token", b"\x00", tokenizer.vocab[0]),
        print_check("first merge preserves order", (b" ", b"t"), tokenizer.merges[0]),
        print_check("second merge preserves order", (b" ", b"a"), tokenizer.merges[1]),
        print_check("third merge preserves order", (b"h", b"e"), tokenizer.merges[2]),
        print_check("special token id 256", b"<|endoftext|>", tokenizer.vocab[256]),
    ]
    return all(checks)


def check_decode() -> bool:
    ascii_tokenizer = Tokenizer({0: b"h", 1: b"i"}, [], [])
    unicode_tokenizer = Tokenizer({0: "é".encode("utf-8")}, [], [])

    checks = [
        print_check("decode ascii bytes", "hi", ascii_tokenizer.decode([0, 1])),
        print_check("decode utf-8 bytes", "é", unicode_tokenizer.decode([0])),
    ]
    return all(checks)


def main() -> int:
    passed = 0
    total = 0

    print(title("Tokenizer Init Sanity Checks"))
    passed += check_init_stores_fields()
    total += 1

    print(title("\nTokenizer from_files Sanity Checks"))
    try:
        passed += check_from_files_loads_artifacts()
    except FileNotFoundError as exc:
        print(warn_text(f"SKIP from_files artifact check: {exc}"))
    else:
        total += 1

    print(title("\nTokenizer Decode Sanity Checks"))
    passed += check_decode()
    total += 1

    if passed == total:
        print(pass_text(f"\nSummary: {passed}/{total} passed"))
        return 0

    print(warn_text(f"\nSummary: {passed}/{total} passed"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
