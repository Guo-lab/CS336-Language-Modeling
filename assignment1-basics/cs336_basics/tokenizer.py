import os
from collections.abc import Iterable, Sequence

Pair = tuple[bytes, bytes]
Pretoken = tuple[bytes, ...]
Vocabulary = dict[int, bytes]
Merges = list[Pair]

PAT = r"""..."""


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: Sequence[str],
) -> tuple[Vocabulary, Merges]:
    raise NotImplementedError


def pretokenize(text: str, special_tokens: Sequence[str]) -> list[Pretoken]:
    raise NotImplementedError


def count_pairs(words: Iterable[Pretoken]) -> dict[Pair, int]:
    raise NotImplementedError


def merge_pair(words: Iterable[Pretoken], pair: Pair) -> list[Pretoken]:
    raise NotImplementedError


class Tokenizer:
    def __init__(
        self,
        vocab: Vocabulary,
        merges: Sequence[Pair],
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        raise NotImplementedError
