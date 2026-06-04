import os
from collections.abc import Iterable
from os import PathLike
from typing import BinaryIO

DEFAULT_TARGET_CHUNK_BYTES = 64 * 1024 * 1024
ChunkRange = tuple[int, int]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    if not isinstance(split_special_token, bytes):
        raise TypeError("split_special_token must be bytes")

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for boundary_index in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[boundary_index]
        # Start at boundary guess
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[boundary_index] = initial_position + found_at
                break

            initial_position += mini_chunk_size
    # Make sure all boundaries are unique,
    # but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def iter_chunk_ranges(
    input_path: str | PathLike[str],
    split_special_token: bytes | None,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
) -> Iterable[ChunkRange]:
    """
    Yield byte ranges that are safe to process independently.

    The returned ranges can be sent to worker processes for independent
    pretoken counting.
    """
    if split_special_token is None:
        # Without a special-token boundary, byte chunking can split a regex pre-token
        # and change training counts, so fall back to one full-text chunk.
        yield 0, os.path.getsize(input_path)
        return

    file_size = os.path.getsize(input_path)
    desired_num_chunks = max(1, (file_size + target_chunk_bytes - 1) // target_chunk_bytes)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token)

    yield from zip(boundaries[:-1], boundaries[1:])


def read_text_chunk(input_path: str | PathLike[str], start: int, end: int) -> str:
    """Read and decode one byte range from a text file."""
    with open(input_path, "rb") as f:
        f.seek(start)
        return f.read(end - start).decode("utf-8", errors="ignore")


# Serial wrapper used when CS336_BPE_NUM_WORKERS is unset or set to 1.
def iter_text_chunks(
    input_path: str | PathLike[str],
    split_special_token: bytes | None,
    target_chunk_bytes: int = DEFAULT_TARGET_CHUNK_BYTES,
) -> Iterable[str]:
    """
    Yield decoded text chunks, splitting on a special-token boundary when provided.
    Serial wrapper: ranges -> text chunks
    """
    # Can be parallelized by sending ranges from iter_chunk_ranges()
    # to worker processes.
    for start, end in iter_chunk_ranges(
        input_path, split_special_token, target_chunk_bytes
    ):
        yield read_text_chunk(input_path, start, end)
