from collections.abc import Iterable, Sequence, Iterator
import json
import re

from cs336_basics.bpe import Merges, Pair, Vocabulary, PAT_RE


class Tokenizer:
    # ===================================
    #        Initialization
    # ===================================
    def __init__(
        self,
        vocab: Vocabulary,
        merges: Merges,
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.vocab = vocab  # id -> bytes
        self.merges = merges  # [(bytes, bytes), ...]
        self.special_tokens = special_tokens  # list[str]

        # Derived lookup tables belong in __init__ so every construction path
        # (direct constructor or from_files) produces a fully usable tokenizer.
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        # Reduce O(number_of_merges) to O(1)
        self.merges_rank = {pair: rank for rank, pair in enumerate(merges)}

        # Speedup BPE encoding
        self.encode_cache: dict[str, list[int]] = {}
        # Larger caches showed little additional benefit.
        self.encode_cache_max_size = 150_000

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
        """
        vocab: Vocabulary = {}
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
            for token_id, each_token in raw_vocab.items():
                token_bytes = bytes.fromhex(each_token["hex"])
                vocab[int(token_id)] = token_bytes

        merges: Merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            raw_merges = json.load(f)
            for left, right in raw_merges:
                left_bytes = bytes.fromhex(left["hex"])
                right_bytes = bytes.fromhex(right["hex"])
                merges.append((left_bytes, right_bytes))

        return cls(vocab, merges, special_tokens)

    # ===================================
    #            Encoding
    # ===================================
    def _split_special_tokens(self, text: str) -> Iterator[tuple[str, bool]]:
        """Yield (piece, is_special). Special tokens are kept as their own pieces."""
        if not self.special_tokens:
            yield text, False
            return

        # ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
        # If special tokens overlap, match longer tokens first.
        special_pattern = "|".join(
            re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)
        )
        pattern: str = f"({special_pattern})"

        special_set = set(self.special_tokens)
        for part in re.split(pattern, text):
            if part == "":
                continue
            yield part, (part in special_set)

    def _best_pair(self, pieces: list[bytes]) -> Pair | None:
        """O(len(pieces)^2)"""
        best_pair: Pair | None = None
        best_rank = len(self.merges_rank)

        for i in range(len(pieces) - 1):
            pair = (pieces[i], pieces[i + 1])
            rank = self.merges_rank.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = pair

        return best_pair

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """
        Byte-level tokenizer with BPE merges
        """
        cached_ids = self.encode_cache.get(pretoken)
        if cached_ids is not None:
            return cached_ids

        token_bytes: bytes = pretoken.encode("utf-8")
        pieces: list[bytes] = [bytes([b]) for b in token_bytes]

        while len(pieces) > 1:
            best_pair = self._best_pair(pieces)
            if best_pair is None:
                break

            merged_pieces: list[bytes] = []
            i = 0
            while i < len(pieces):
                if i < len(pieces) - 1 and (pieces[i], pieces[i + 1]) == best_pair:
                    merged_pieces.append(pieces[i] + pieces[i + 1])
                    i += 2
                else:
                    merged_pieces.append(pieces[i])
                    i += 1
            pieces = merged_pieces

        token_ids = [self.token_to_id[piece] for piece in pieces]
        if len(self.encode_cache) >= self.encode_cache_max_size:
            self.encode_cache.clear()
        self.encode_cache[pretoken] = token_ids
        return token_ids

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        token_ids: list[int] = []

        for piece, is_special in self._split_special_tokens(text):
            if is_special:
                token_ids.append(self.token_to_id[piece.encode("utf-8")])
                continue

            for match in PAT_RE.finditer(piece):
                pretoken: str = match.group()
                token_ids.extend(self._encode_pretoken(pretoken))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator
        that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we
        cannot directly load into memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    # ===================================
    #            Decoding
    # ===================================
    def decode(self, ids: Iterable[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens_bytes: list[bytes] = []
        # Decode a sequence of token IDs by concatenating their bytes in order.
        # Spaces, newlines, and null bytes are preserved if present.
        for token_id in ids:
            token_bytes: bytes = self.vocab[token_id]
            tokens_bytes.append(token_bytes)
        text_bytes = b"".join(tokens_bytes)
        return text_bytes.decode("utf-8", errors="replace")
