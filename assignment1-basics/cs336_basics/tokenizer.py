from collections.abc import Iterable, Sequence

from cs336_basics.bpe import Merges, Pair, Vocabulary


class Tokenizer:
    def __init__(
        self,
        vocab: Vocabulary,
        merges: Merges,
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Sequence[str] | None = None,
    ) -> "Tokenizer":
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output) and
        (optionally) a list of special tokens.
        This method should accept the following additional parameters.
        """
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        raise NotImplementedError

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator
        that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we
        cannot directly load into memory.
        """
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        """Decode a sequence of token IDs into text."""
        raise NotImplementedError
