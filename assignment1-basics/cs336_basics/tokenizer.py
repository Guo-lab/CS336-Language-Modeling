from collections.abc import Iterable, Sequence

from cs336_basics.bpe import Merges, Pair, Vocabulary


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
