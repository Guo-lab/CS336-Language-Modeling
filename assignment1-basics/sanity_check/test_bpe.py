"""Manual sanity checks for cs336_basics.bpe.

Run from the assignment root:
    python sanity_check/test_bpe.py
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.bpe import (  # noqa: E402
    Pair,
    Pretoken,
    PretokenCounts,
    Vocabulary,
    build_pretoken_counts,
    build_initial_vocab,
    count_pairs,
    count_pairs_from_pretoken_counts,
    merge_pair,
    merge_pretoken_counts,
    pretokenize,
    split_text_by_special_tokens,
    train_bpe,
)
from sanity_check.utils import fail_text, pass_text, title, warn_text  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    expected: list[Pretoken]


@dataclass(frozen=True)
class PairCountCase:
    name: str
    words: list[Pretoken]
    expected: dict[Pair, int]


@dataclass(frozen=True)
class PretokenCountCase:
    name: str
    chunks: list[str]
    expected: PretokenCounts


@dataclass(frozen=True)
class WeightedPairCountCase:
    name: str
    pretoken_counts: PretokenCounts
    expected: dict[Pair, int]


@dataclass(frozen=True)
class MergePretokenCountCase:
    name: str
    pretoken_counts: PretokenCounts
    pair: Pair
    expected: PretokenCounts


@dataclass(frozen=True)
class MergePairCase:
    name: str
    words: list[Pretoken]
    pair: Pair
    expected: list[Pretoken]


@dataclass(frozen=True)
class InitialVocabCase:
    name: str
    special_tokens: list[str]
    expected_items: dict[int, bytes]
    expected_size: int


@dataclass(frozen=True)
class SpecialSplitCase:
    name: str
    text: str
    special_tokens: list[str]
    expected: list[str]


@dataclass(frozen=True)
class TrainBpeCase:
    name: str
    text: str
    vocab_size: int
    special_tokens: list[str]
    expected_merges: list[Pair]
    expected_vocab_items: dict[int, bytes]


def as_pretoken(text: str) -> Pretoken:
    return tuple(bytes([byte]) for byte in text.encode("utf-8"))


def flatten(pretokens: list[Pretoken]) -> bytes:
    return b"".join(b"".join(pretoken) for pretoken in pretokens)


def print_case(case: Case) -> bool:
    print(title(f"\n== {case.name} =="))
    print(f"text: {case.text!r}")
    print(f"expected: {case.expected!r}")

    try:
        result = pretokenize(case.text)
    except Exception as exc:
        print(fail_text(f"FAIL: raised {type(exc).__name__}: {exc}"))
        return False

    print(f"result:   {result!r}")
    print(f"expected flattened: {flatten(case.expected)!r}")
    print(f"result flattened:   {flatten(result)!r}")

    if result == case.expected:
        print(pass_text())
        return True

    print(fail_text("FAIL: result does not match expected"))
    return False


def print_pair_count_case(case: PairCountCase) -> bool:
    print(title(f"\n== {case.name} =="))
    print(f"words: {case.words!r}")
    print(f"expected: {case.expected!r}")

    try:
        result = count_pairs(case.words)
    except Exception as exc:
        print(fail_text(f"FAIL: raised {type(exc).__name__}: {exc}"))
        return False

    print(f"result:   {result!r}")

    if result == case.expected:
        print(pass_text())
        return True

    print(fail_text("FAIL: result does not match expected"))
    return False


def print_pretoken_count_case(case: PretokenCountCase) -> bool:
    print(title(f"\n== {case.name} =="))
    print(f"chunks: {case.chunks!r}")
    print(f"expected: {case.expected!r}")

    try:
        result = build_pretoken_counts(case.chunks)
    except Exception as exc:
        print(fail_text(f"FAIL: raised {type(exc).__name__}: {exc}"))
        return False

    print(f"result:   {result!r}")

    if result == case.expected:
        print(pass_text())
        return True

    print(fail_text("FAIL: result does not match expected"))
    return False


def print_weighted_pair_count_case(case: WeightedPairCountCase) -> bool:
    print(title(f"\n== {case.name} =="))
    print(f"pretoken_counts: {case.pretoken_counts!r}")
    print(f"expected: {case.expected!r}")

    try:
        result = count_pairs_from_pretoken_counts(case.pretoken_counts)
    except Exception as exc:
        print(fail_text(f"FAIL: raised {type(exc).__name__}: {exc}"))
        return False

    print(f"result:   {result!r}")

    if result == case.expected:
        print(pass_text())
        return True

    print(fail_text("FAIL: result does not match expected"))
    return False


def print_merge_pretoken_count_case(case: MergePretokenCountCase) -> bool:
    print(title(f"\n== {case.name} =="))
    print(f"pretoken_counts: {case.pretoken_counts!r}")
    print(f"pair: {case.pair!r}")
    print(f"expected: {case.expected!r}")

    try:
        result = merge_pretoken_counts(case.pretoken_counts, case.pair)
    except Exception as exc:
        print(fail_text(f"FAIL: raised {type(exc).__name__}: {exc}"))
        return False

    print(f"result:   {result!r}")

    if result == case.expected:
        print(pass_text())
        return True

    print(fail_text("FAIL: result does not match expected"))
    return False


def print_merge_pair_case(case: MergePairCase) -> bool:
    print(title(f"\n== {case.name} =="))
    print(f"words: {case.words!r}")
    print(f"pair: {case.pair!r}")
    print(f"expected: {case.expected!r}")

    try:
        result = merge_pair(case.words, case.pair)
    except Exception as exc:
        print(fail_text(f"FAIL: raised {type(exc).__name__}: {exc}"))
        return False

    print(f"result:   {result!r}")

    if result == case.expected:
        print(pass_text())
        return True

    print(fail_text("FAIL: result does not match expected"))
    return False


def print_initial_vocab_case(case: InitialVocabCase) -> bool:
    print(title(f"\n== {case.name} =="))
    print(f"special_tokens: {case.special_tokens!r}")
    print(f"expected_size: {case.expected_size!r}")
    print(f"expected_items: {case.expected_items!r}")

    try:
        result: Vocabulary = build_initial_vocab(case.special_tokens)
    except Exception as exc:
        print(fail_text(f"FAIL: raised {type(exc).__name__}: {exc}"))
        return False

    result_items = {key: result.get(key) for key in case.expected_items}
    print(f"result_size: {len(result)!r}")
    print(f"result_items: {result_items!r}")

    if len(result) == case.expected_size and result_items == case.expected_items:
        print(pass_text())
        return True

    print(fail_text("FAIL: vocab does not match expected"))
    return False


def print_special_split_case(case: SpecialSplitCase) -> bool:
    print(title(f"\n== {case.name} =="))
    print(f"text: {case.text!r}")
    print(f"special_tokens: {case.special_tokens!r}")
    print(f"expected: {case.expected!r}")

    try:
        result = split_text_by_special_tokens(case.text, case.special_tokens)
    except Exception as exc:
        print(fail_text(f"FAIL: raised {type(exc).__name__}: {exc}"))
        return False

    print(f"result:   {result!r}")

    if result == case.expected:
        print(pass_text())
        return True

    print(fail_text("FAIL: result does not match expected"))
    return False


def print_train_bpe_case(case: TrainBpeCase) -> bool:
    print(title(f"\n== {case.name} =="))
    print(f"text: {case.text!r}")
    print(f"vocab_size: {case.vocab_size!r}")
    print(f"special_tokens: {case.special_tokens!r}")
    print(f"expected_merges: {case.expected_merges!r}")
    print(f"expected_vocab_items: {case.expected_vocab_items!r}")

    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(case.text)
            input_path = f.name

        vocab, merges = train_bpe(input_path, case.vocab_size, case.special_tokens)
    except Exception as exc:
        print(fail_text(f"FAIL: raised {type(exc).__name__}: {exc}"))
        return False
    finally:
        if "input_path" in locals():
            Path(input_path).unlink(missing_ok=True)

    result_vocab_items = {key: vocab.get(key) for key in case.expected_vocab_items}
    print(f"result_merges: {merges!r}")
    print(f"result_vocab_items: {result_vocab_items!r}")

    if merges == case.expected_merges and result_vocab_items == case.expected_vocab_items:
        print(pass_text())
        return True

    print(fail_text("FAIL: train_bpe result does not match expected"))
    return False


def main() -> int:
    initial_vocab_cases = [
        InitialVocabCase(
            "byte vocabulary without specials",
            [],
            {0: b"\x00", 65: b"A", 255: b"\xff"},
            256,
        ),
        InitialVocabCase(
            "append special tokens after bytes",
            ["<|endoftext|>", "<pad>"],
            {256: b"<|endoftext|>", 257: b"<pad>"},
            258,
        ),
    ]

    print(title("Initial Vocab Sanity Checks"))
    passed = sum(print_initial_vocab_case(case) for case in initial_vocab_cases)
    total = len(initial_vocab_cases)

    special_split_cases = [
        SpecialSplitCase("no special tokens", "hello", [], ["hello"]),
        SpecialSplitCase(
            "single boundary",
            "hi<|endoftext|>there",
            ["<|endoftext|>"],
            ["hi", "there"],
        ),
        SpecialSplitCase(
            "repeated boundary",
            "a<|endoftext|><|endoftext|>b",
            ["<|endoftext|>"],
            ["a", "b"],
        ),
        SpecialSplitCase(
            "leading and trailing boundary",
            "<|endoftext|>middle<|endoftext|>",
            ["<|endoftext|>"],
            ["middle"],
        ),
        SpecialSplitCase(
            "multiple special tokens",
            "a<pad>b<|endoftext|>c",
            ["<|endoftext|>", "<pad>"],
            ["a", "b", "c"],
        ),
    ]

    print(title("\nSpecial Token Split Sanity Checks"))
    passed += sum(print_special_split_case(case) for case in special_split_cases)
    total += len(special_split_cases)

    pretokenize_cases = [
        Case("basic ascii", "hi!", [as_pretoken("hi"), as_pretoken("!")]),
        Case(
            "space and punctuation",
            "Hello, world!",
            [as_pretoken("Hello"), as_pretoken(","), as_pretoken(" world"), as_pretoken("!")],
        ),
        Case("utf-8 bytes", "cafe é", [as_pretoken("cafe"), as_pretoken(" é")]),
    ]

    print(title("Pretokenize Sanity Checks"))
    passed += sum(print_case(case) for case in pretokenize_cases)
    total += len(pretokenize_cases)

    pair_count_cases = [
        PairCountCase(
            "count adjacent pairs",
            [as_pretoken("banana")],
            {
                (b"b", b"a"): 1,
                (b"a", b"n"): 2,
                (b"n", b"a"): 2,
            },
        ),
        PairCountCase(
            "do not count across pretokens",
            [as_pretoken("ab"), as_pretoken("cd")],
            {
                (b"a", b"b"): 1,
                (b"c", b"d"): 1,
            },
        ),
        PairCountCase("short pretokens have no pairs", [(), (b"x",)], {}),
    ]

    print(title("\nPair Count Sanity Checks"))
    passed += sum(print_pair_count_case(case) for case in pair_count_cases)
    total += len(pair_count_cases)

    pretoken_count_cases = [
        PretokenCountCase(
            "count repeated pretokens across chunks",
            ["ab ab", "ab"],
            {
                as_pretoken("ab"): 2,
                as_pretoken(" ab"): 1,
            },
        ),
        PretokenCountCase("empty chunks produce no counts", ["", ""], {}),
    ]

    print(title("\nPretoken Count Sanity Checks"))
    passed += sum(print_pretoken_count_case(case) for case in pretoken_count_cases)
    total += len(pretoken_count_cases)

    weighted_pair_count_cases = [
        WeightedPairCountCase(
            "weight pair counts by pretoken frequency",
            {
                as_pretoken("ab"): 3,
                as_pretoken("ac"): 2,
            },
            {
                (b"a", b"b"): 3,
                (b"a", b"c"): 2,
            },
        ),
        WeightedPairCountCase(
            "skip short counted pretokens",
            {
                (): 5,
                (b"x",): 4,
            },
            {},
        ),
    ]

    print(title("\nWeighted Pair Count Sanity Checks"))
    passed += sum(print_weighted_pair_count_case(case) for case in weighted_pair_count_cases)
    total += len(weighted_pair_count_cases)

    merge_pair_cases = [
        MergePairCase(
            "merge one pair",
            [as_pretoken("abc")],
            (b"a", b"b"),
            [(b"ab", b"c")],
        ),
        MergePairCase(
            "merge multiple pretokens",
            [as_pretoken("ab"), as_pretoken("cab")],
            (b"a", b"b"),
            [(b"ab",), (b"c", b"ab")],
        ),
        MergePairCase(
            "left-to-right non-overlapping merge",
            [as_pretoken("aaa")],
            (b"a", b"a"),
            [(b"aa", b"a")],
        ),
        MergePairCase(
            "no matching pair leaves words unchanged",
            [as_pretoken("abc")],
            (b"x", b"y"),
            [as_pretoken("abc")],
        ),
    ]

    print(title("\nMerge Pair Sanity Checks"))
    passed += sum(print_merge_pair_case(case) for case in merge_pair_cases)
    total += len(merge_pair_cases)

    merge_pretoken_count_cases = [
        MergePretokenCountCase(
            "merge counted pretokens",
            {
                as_pretoken("ab"): 2,
                as_pretoken("cab"): 1,
            },
            (b"a", b"b"),
            {
                (b"ab",): 2,
                (b"c", b"ab"): 1,
            },
        ),
        MergePretokenCountCase(
            "combine counts when merges collide",
            {
                as_pretoken("ab"): 2,
                (b"ab",): 3,
            },
            (b"a", b"b"),
            {
                (b"ab",): 5,
            },
        ),
    ]

    print(title("\nMerge Pretoken Count Sanity Checks"))
    passed += sum(print_merge_pretoken_count_case(case) for case in merge_pretoken_count_cases)
    total += len(merge_pretoken_count_cases)

    train_bpe_cases = [
        TrainBpeCase(
            "learn one obvious merge",
            "abab",
            257,
            [],
            [(b"a", b"b")],
            {256: b"ab"},
        ),
        TrainBpeCase(
            "special token is vocab only",
            "ab<|endoftext|>ab",
            258,
            ["<|endoftext|>"],
            [(b"a", b"b")],
            {256: b"<|endoftext|>", 257: b"ab"},
        ),
        TrainBpeCase(
            "stop when no pairs exist",
            "!",
            258,
            [],
            [],
            {},
        ),
        TrainBpeCase(
            "tie chooses lexicographically largest pair",
            "abac",
            257,
            [],
            [(b"b", b"a")],
            {256: b"ba"},
        ),
    ]

    print(title("\nTrain BPE Sanity Checks"))
    passed += sum(print_train_bpe_case(case) for case in train_bpe_cases)
    total += len(train_bpe_cases)

    if passed == total:
        print(pass_text(f"\nSummary: {passed}/{total} passed"))
        return 0

    print(warn_text(f"\nSummary: {passed}/{total} passed"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
