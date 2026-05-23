import os
from collections.abc import Iterable, Sequence

import regex

Pair = tuple[bytes, bytes]
Pretoken = tuple[bytes, ...]  # one regex token represented as a sequence of byte-symbols
Vocabulary = dict[int, bytes]
Merges = list[Pair]


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def build_initial_vocab(special_tokens: Sequence[str]) -> Vocabulary:
    """Build the initial byte vocabulary, then append special tokens in order."""
    vocab: Vocabulary = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    return vocab


def split_text_by_special_tokens(text: str, special_tokens: Sequence[str]) -> list[str]:
    """Split text on special tokens, dropping the special tokens and empty chunks."""
    if not special_tokens:
        return [text] if text else []

    escaped_special_tokens = [
        regex.escape(special_token)
        for special_token in sorted(special_tokens, key=len, reverse=True)
    ]
    special_token_pattern = "|".join(escaped_special_tokens)
    return [chunk for chunk in regex.split(special_token_pattern, text) if chunk]


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: Sequence[str],
) -> tuple[Vocabulary, Merges]:
    with open(input_path, encoding="utf-8") as f:
        corpus = f.read()
    chunks = split_text_by_special_tokens(corpus, special_tokens)

    vocab = build_initial_vocab(special_tokens)
    merges: Merges = []

    pretokens = []
    for chunk in chunks:
        pretokens.extend(pretokenize(chunk))

    while len(vocab) < vocab_size:
        counts = count_pairs(pretokens)
        if not counts:
            break
        best_pair = max(counts, key=lambda candidate: (counts[candidate], candidate))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        pretokens = merge_pair(pretokens, best_pair)

    return vocab, merges


def pretokenize(text: str) -> list[Pretoken]:
    """Split ordinary text into regex pre-tokens, then byte-level BPE symbols."""
    # Special-token boundaries are handled by the caller outside this helper,
    # noted in the PDF; this function should only see ordinary text chunks.
    pretokens_in_str = regex.findall(PAT, text)

    # Each regex match is one string pre-token. BPE starts from byte-level
    # symbols, so each UTF-8 byte becomes its own bytes object.
    pretokens_in_bytes = [pretoken_in_str.encode("utf-8") for pretoken_in_str in pretokens_in_str]

    pretokens = []
    for each_pretoken_in_bytes in pretokens_in_bytes:
        pretoken = tuple(bytes([byte]) for byte in each_pretoken_in_bytes)
        pretokens.append(pretoken)
    return pretokens


def count_pairs(words: Iterable[Pretoken]) -> dict[Pair, int]:
    """Count adjacent symbols inside each pretoken; never across pretoken boundaries."""
    counts: dict[Pair, int] = {}
    for pretoken in words:
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(words: Iterable[Pretoken], pair: Pair) -> list[Pretoken]:
    """Merge one pair inside each pretoken, left-to-right and non-overlapping."""
    updated_pretokens = []
    for pretoken in words:
        # Temporary mutable container for the bytes symbols in one updated pretoken.
        updated_symbols = []
        i = 0
        while i < len(pretoken):
            if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == pair:
                updated_symbols.append(pretoken[i] + pretoken[i + 1])
                i += 2
            else:
                updated_symbols.append(pretoken[i])
                i += 1
        updated_pretokens.append(tuple(updated_symbols))
    return updated_pretokens
