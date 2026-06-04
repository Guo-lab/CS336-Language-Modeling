import os
import regex
from collections.abc import Iterable, Sequence
from heapq import heapify, heappop, heappush
from multiprocessing import Pool

from cs336_basics.chunking import (
    DEFAULT_TARGET_CHUNK_BYTES,
    iter_chunk_ranges,
    iter_text_chunks,
    read_text_chunk,
)

Pretoken = tuple[bytes, ...]  # one regex token represented as a sequence of byte-symbols
PretokenCounts = dict[Pretoken, int]
PretokenCountTask = tuple[str, int, int, tuple[str, ...]]  # (file, start, end, special_tokens)

Pair = tuple[bytes, bytes]
PairCounts = dict[Pair, int]
PairIndex = dict[Pair, set[Pretoken]]
PairHeapEntry = tuple[int, "_MaxPairKey", Pair]

Vocabulary = dict[int, bytes]
Merges = list[Pair]


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = regex.compile(PAT)


class _MaxPairKey:
    """Heap key that orders lexicographically larger pairs first."""

    def __init__(self, pair: Pair) -> None:
        self.pair = pair

    def __lt__(self, other: "_MaxPairKey") -> bool:
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MaxPairKey) and self.pair == other.pair


def count_pairs_in_pretoken(pretoken: Pretoken) -> PairCounts:
    """Count adjacent pair occurrences inside one pretoken."""
    counts: PairCounts = {}
    for i in range(len(pretoken) - 1):
        pair = pretoken[i], pretoken[i + 1]
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def add_pretoken_count(
    pretoken_counts: PretokenCounts,
    pretoken: Pretoken,
    count: int,
) -> None:
    """Add count to one pretoken entry, creating it if needed."""
    pretoken_counts[pretoken] = pretoken_counts.get(pretoken, 0) + count


def add_pretoken_counts(dst: PretokenCounts, src: PretokenCounts) -> None:
    """Add all pretoken counts from src into dst."""
    for pretoken, count in src.items():
        add_pretoken_count(dst, pretoken, count)


def build_pretoken_counts_for_range(task: PretokenCountTask) -> PretokenCounts:
    """Read one file range and count pretokens in that range."""
    input_path, start, end, special_tokens = task
    text = read_text_chunk(input_path, start, end)
    chunks = BPETrainer.split_text_by_special_tokens(text, special_tokens)
    return BPETrainer.build_pretoken_counts(chunks)


class PairState:
    """Pair counts plus the pretokens that currently contain each pair."""

    HEAP_REBUILD_RATIO = 8

    def __init__(self, pretoken_counts: PretokenCounts) -> None:
        self.counts: PairCounts = {}
        self.index: PairIndex = {}
        self.heap: list[PairHeapEntry] = []

        for pretoken, pretoken_count in pretoken_counts.items():
            self.add(pretoken, pretoken_count)

    def __bool__(self) -> bool:
        return bool(self.counts)

    def best_pair(self) -> Pair:
        self.maybe_rebuild_heap()
        while self.heap:
            # Lazy invalidation: The heap may contain stale entries because pair counts are updated
            # by pushing new entries instead of deleting old ones. Discard popped entries whose count
            # no longer matches self.counts.
            neg_count, _, pair = heappop(self.heap)
            count = -neg_count
            if self.counts.get(pair) == count:
                return pair
        raise ValueError("Cannot select a best pair from empty pair state")

    def affected_pretokens(self, pair: Pair) -> list[Pretoken]:
        return list(self.index.get(pair, ()))

    def add(self, pretoken: Pretoken, pretoken_count: int) -> None:
        for pair, pair_count in count_pairs_in_pretoken(pretoken).items():
            self.counts[pair] = self.counts.get(pair, 0) + pair_count * pretoken_count
            self.index.setdefault(pair, set()).add(pretoken)
            self.push_pair(pair)

    def remove(self, pretoken: Pretoken, pretoken_count: int) -> None:
        for pair, pair_count in count_pairs_in_pretoken(pretoken).items():
            updated_count = self.counts[pair] - pair_count * pretoken_count
            if updated_count > 0:
                self.counts[pair] = updated_count
                self.push_pair(pair)
            else:
                del self.counts[pair]

            pretokens = self.index[pair]
            pretokens.discard(pretoken)
            if not pretokens:
                del self.index[pair]

    def push_pair(self, pair: Pair) -> None:
        """The Min-Heap pops the largest count first, using lexicographic order as a tie-break."""
        heappush(self.heap, (-self.counts[pair], _MaxPairKey(pair), pair))

    def maybe_rebuild_heap(self) -> None:
        """Compact stale heap entries when lazy invalidation has grown too large."""
        if not self.counts:
            self.heap.clear()
            return

        if len(self.heap) <= self.HEAP_REBUILD_RATIO * len(self.counts):
            return

        self.heap = [(-count, _MaxPairKey(pair), pair) for pair, count in self.counts.items()]
        heapify(self.heap)


class BPETrainer:
    """Train a byte-level BPE vocabulary from ordinary text chunks."""

    def __init__(self, vocab_size: int, special_tokens: Sequence[str]) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = tuple(special_tokens)

    @staticmethod
    def build_initial_vocab(special_tokens: Sequence[str]) -> Vocabulary:
        """Build the initial byte vocabulary, then append special tokens in order."""
        vocab: Vocabulary = {i: bytes([i]) for i in range(256)}

        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")

        return vocab

    @staticmethod
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

    def train(self, input_path: str | os.PathLike[str]) -> tuple[Vocabulary, Merges]:
        pretoken_counts = self.build_pretoken_counts_from_file(input_path)

        # Speed key: build pair counts once, plus an inverted index from each
        # pair to the pretokens containing it, so each merge updates only the
        # affected pretokens instead of rescanning the whole corpus.
        pair_state = PairState(pretoken_counts)

        vocab = self.build_initial_vocab(self.special_tokens)
        merges: Merges = []

        while len(vocab) < self.vocab_size:
            if not pair_state:
                break

            best_pair = pair_state.best_pair()
            merges.append(best_pair)
            vocab[len(vocab)] = best_pair[0] + best_pair[1]
            self.merge_pair_in_state(pretoken_counts, pair_state, best_pair)

        return vocab, merges

    def build_pretoken_counts_from_file(
        self, input_path: str | os.PathLike[str]
    ) -> PretokenCounts:
        """Count pretokens from a file, chunking on special-token boundaries."""
        target_chunk_bytes = int(
            os.environ.get("CS336_BPE_CHUNK_BYTES", str(DEFAULT_TARGET_CHUNK_BYTES))
        )
        num_workers = int(os.environ.get("CS336_BPE_NUM_WORKERS", "1"))
        split_special_token = self.split_special_token_bytes()

        if num_workers > 1:
            return self.build_pretoken_counts_parallel(
                input_path, split_special_token, target_chunk_bytes, num_workers
            )

        return self.build_pretoken_counts_serial(
            input_path, split_special_token, target_chunk_bytes
        )

    def split_special_token_bytes(self) -> bytes | None:
        """Return the special token used for chunk boundaries, if any."""
        if not self.special_tokens:
            return None
        return max(self.special_tokens, key=len).encode("utf-8")

    def build_pretoken_counts_serial(
        self,
        input_path: str | os.PathLike[str],
        split_special_token: bytes | None,
        target_chunk_bytes: int,
    ) -> PretokenCounts:
        """Count pretokens from file chunks in the current process."""
        pretoken_counts: PretokenCounts = {}
        for text in iter_text_chunks(input_path, split_special_token, target_chunk_bytes):
            chunks = self.split_text_by_special_tokens(text, self.special_tokens)
            add_pretoken_counts(pretoken_counts, self.build_pretoken_counts(chunks))

        return pretoken_counts

    def build_pretoken_counts_parallel(
        self,
        input_path: str | os.PathLike[str],
        split_special_token: bytes | None,
        target_chunk_bytes: int,
        num_workers: int,
    ) -> PretokenCounts:
        """Count pretokens from file chunks across worker processes."""
        ranges = iter_chunk_ranges(input_path, split_special_token, target_chunk_bytes)
        tasks = (
            (os.fspath(input_path), start, end, self.special_tokens) for start, end in ranges
        )

        pretoken_counts: PretokenCounts = {}
        with Pool(processes=num_workers) as pool:
            for chunk_counts in pool.imap_unordered(build_pretoken_counts_for_range, tasks):
                add_pretoken_counts(pretoken_counts, chunk_counts)

        return pretoken_counts

    # ===================================
    #           Pretokenization
    # ===================================
    @staticmethod
    def iter_pretokens(text: str) -> Iterable[Pretoken]:
        """Yield regex pre-tokens as byte-level BPE symbol tuples."""
        # Special-token boundaries are handled by the caller before this helper,
        # as noted in the PDF. This function should only see ordinary text chunks.
        for pretoken_in_str in PAT_RE.findall(text):
            # Each regex match is one string pre-token. BPE starts from byte-level
            # symbols, so each UTF-8 byte becomes its own bytes object.
            pretoken_in_bytes = pretoken_in_str.encode("utf-8")
            yield tuple(bytes([byte]) for byte in pretoken_in_bytes)

    @classmethod
    def pretokenize(cls, text: str) -> list[Pretoken]:
        """Split ordinary text into regex pre-tokens, then byte-level BPE symbols."""
        return list(cls.iter_pretokens(text))

    @classmethod
    def build_pretoken_counts(cls, chunks: Iterable[str]) -> PretokenCounts:
        """Count repeated pretokens after splitting away special-token chunks."""
        pretoken_counts: PretokenCounts = {}
        for chunk in chunks:
            for pretoken in cls.iter_pretokens(chunk):
                add_pretoken_count(pretoken_counts, pretoken, 1)
        return pretoken_counts

    # ===================================
    #           Pair Counting
    # ===================================
    # Core pair-counting logic is abstracted above and reused by PairState.
    # ! not used by train(); kept for sanity checks.
    @staticmethod
    def count_pairs(words: Iterable[Pretoken]) -> PairCounts:
        """Count adjacent symbols inside each pretoken; never across pretoken boundaries."""
        counts: PairCounts = {}
        for pretoken in words:
            for pair, pair_count in count_pairs_in_pretoken(pretoken).items():
                counts[pair] = counts.get(pair, 0) + pair_count
        return counts

    # ! not used by train(); kept for sanity checks.
    @staticmethod
    def count_pairs_from_pretoken_counts(pretoken_counts: PretokenCounts) -> PairCounts:
        """Count adjacent pairs weighted by each pretoken's frequency."""
        counts: PairCounts = {}
        for pretoken, pretoken_count in pretoken_counts.items():
            for pair, pair_count in count_pairs_in_pretoken(pretoken).items():
                counts[pair] = counts.get(pair, 0) + pair_count * pretoken_count
        return counts

    # ===================================
    #           Merging
    # ===================================
    @staticmethod
    def merge_pretoken(pretoken: Pretoken, pair: Pair) -> Pretoken:
        """Merge one pair in one pretoken, left-to-right and non-overlapping."""
        # Build a new tuple only after the first actual merge. Most pretokens do
        # not contain the current best pair in later rounds, so unchanged
        # pretokens can keep their original tuple object.
        updated_symbols: list[bytes] | None = None
        i = 0
        while i < len(pretoken):
            if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == pair:
                if updated_symbols is None:
                    updated_symbols = list(pretoken[:i])
                updated_symbols.append(pretoken[i] + pretoken[i + 1])
                i += 2
            else:
                if updated_symbols is not None:
                    updated_symbols.append(pretoken[i])
                i += 1

        if updated_symbols is None:
            return pretoken
        return tuple(updated_symbols)

    # ! not used by train(); kept for sanity checks.
    @classmethod
    def merge_pair(cls, words: Iterable[Pretoken], pair: Pair) -> list[Pretoken]:
        """Merge one pair inside each pretoken, left-to-right and non-overlapping."""
        return [cls.merge_pretoken(pretoken, pair) for pretoken in words]

    # ! not used by train(); kept for sanity checks.
    @classmethod
    def merge_pretoken_counts(
        cls, pretoken_counts: PretokenCounts, pair: Pair
    ) -> PretokenCounts:
        """Merge a pair in unique pretokens, preserving and combining frequencies."""
        updated_cnts: PretokenCounts = {}
        for pretoken, pretoken_count in pretoken_counts.items():
            merged_pretoken = cls.merge_pretoken(pretoken, pair)
            updated_cnts[merged_pretoken] = (
                updated_cnts.get(merged_pretoken, 0) + pretoken_count
            )
        return updated_cnts

    @classmethod
    def merge_pair_in_state(
        cls,
        pretoken_cnts: PretokenCounts,
        pair_state: PairState,
        pair: Pair,
    ) -> None:
        """Merge one pair and update only pretokens that contain that pair."""
        additions: PretokenCounts = {}

        for pretoken in pair_state.affected_pretokens(pair):
            pretoken_count = pretoken_cnts.pop(pretoken)
            merged_pretoken = cls.merge_pretoken(pretoken, pair)
            pair_state.remove(pretoken, pretoken_count)
            add_pretoken_count(additions, merged_pretoken, pretoken_count)

        # Add merged pretokens after all removals,
        # because several old pretokens can collapse
        # into the same new pretoken.
        for merged_pretoken, added_count in additions.items():
            add_pretoken_count(pretoken_cnts, merged_pretoken, added_count)
            pair_state.add(merged_pretoken, added_count)
