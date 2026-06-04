from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from cs336_basics.tokenizer import Tokenizer


SPECIAL_TOKEN = "<|endoftext|>"


class TerminalStyle:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenize a text dataset into a 1D .npy token array.")
    parser.add_argument("--input", type=Path, required=True, help="Input text file.")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Directory containing vocab.json and merges.json.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .npy token array.")
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "uint32", "int64"])
    parser.add_argument("--special-token", default=SPECIAL_TOKEN)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def iter_text(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8") as f:
        yield from f


def load_tokenizer(tokenizer_dir: Path, special_token: str) -> Tokenizer:
    return Tokenizer.from_files(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merges.json"),
        special_tokens=[special_token],
    )


def validate_dtype(tokenizer: Tokenizer, dtype: np.dtype) -> None:
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"Token dtype must be integer, got {dtype}")

    max_token_id = max(tokenizer.vocab)
    dtype_max = np.iinfo(dtype).max
    if max_token_id > dtype_max:
        raise ValueError(
            f"Tokenizer max token ID {max_token_id} does not fit in dtype {dtype} "
            f"(max {dtype_max})"
        )


def write_token_buffer(
    tokenizer: Tokenizer,
    input_path: Path,
    raw_path: Path,
    dtype: np.dtype,
    buffer_size: int,
) -> int:
    token_count = 0
    buffer: list[int] = []

    with raw_path.open("wb") as f:
        for token_id in tokenizer.encode_iterable(iter_text(input_path)):
            buffer.append(token_id)
            if len(buffer) >= buffer_size:
                token_count += flush_buffer(buffer, f, dtype)

        token_count += flush_buffer(buffer, f, dtype)

    return token_count


def flush_buffer(buffer: list[int], file, dtype: np.dtype) -> int:
    if not buffer:
        return 0

    arr = np.asarray(buffer, dtype=dtype)
    arr.tofile(file)
    written = len(buffer)
    buffer.clear()
    return written


def raw_to_npy(raw_path: Path, output_path: Path, dtype: np.dtype, token_count: int) -> None:
    raw_tokens = np.memmap(raw_path, mode="r", dtype=dtype, shape=(token_count,))
    output_tokens = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=(token_count,),
    )
    output_tokens[:] = raw_tokens[:]
    output_tokens.flush()


def write_summary(
    path: Path,
    input_path: Path,
    tokenizer_dir: Path,
    output_path: Path,
    dtype: np.dtype,
    token_count: int,
    elapsed_seconds: float,
) -> None:
    summary = {
        "input": str(input_path),
        "tokenizer": str(tokenizer_dir),
        "output": str(output_path),
        "dtype": str(dtype),
        "tokens": token_count,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": token_count / elapsed_seconds if elapsed_seconds > 0 else None,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = build_parser().parse_args()
    dtype = np.dtype(args.dtype)

    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(args.output.suffix + ".tmp.bin")
    summary_path = args.output.with_suffix(".summary.json")

    print(f"{TerminalStyle.BOLD}{TerminalStyle.CYAN}Tokenize dataset{TerminalStyle.RESET}")
    print(f"{TerminalStyle.GREEN}input:{TerminalStyle.RESET} {args.input}")
    print(f"{TerminalStyle.GREEN}tokenizer:{TerminalStyle.RESET} {args.tokenizer}")
    print(f"{TerminalStyle.GREEN}output:{TerminalStyle.RESET} {args.output}")
    print(f"{TerminalStyle.DIM}temporary raw token buffer: {raw_path}{TerminalStyle.RESET}")

    start = time.perf_counter()
    tokenizer = load_tokenizer(args.tokenizer, args.special_token)
    validate_dtype(tokenizer, dtype)
    token_count = write_token_buffer(
        tokenizer=tokenizer,
        input_path=args.input,
        raw_path=raw_path,
        dtype=dtype,
        buffer_size=args.buffer_size,
    )
    raw_to_npy(raw_path, args.output, dtype, token_count)
    elapsed_seconds = time.perf_counter() - start
    raw_path.unlink(missing_ok=True)
    write_summary(
        summary_path,
        input_path=args.input,
        tokenizer_dir=args.tokenizer,
        output_path=args.output,
        dtype=dtype,
        token_count=token_count,
        elapsed_seconds=elapsed_seconds,
    )

    print(f"{TerminalStyle.YELLOW}tokens:{TerminalStyle.RESET} {token_count:,}")
    print(f"{TerminalStyle.YELLOW}elapsed:{TerminalStyle.RESET} {elapsed_seconds:.2f}s")
    print(f"{TerminalStyle.GREEN}summary:{TerminalStyle.RESET} {summary_path}")


if __name__ == "__main__":
    main()
