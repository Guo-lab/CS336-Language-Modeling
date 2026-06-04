from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import platform
import resource
import sys
import threading
import time
from pathlib import Path

import psutil

from cs336_basics.bpe import BPETrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer and save experiment artifacts.")
    parser.add_argument("--input", required=True, type=Path, help="Training text file.")
    parser.add_argument("--vocab-size", required=True, type=int, help="Final vocabulary size, including byte and special tokens.")
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="Special token to add. Repeat this flag for multiple special tokens.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for vocab, merges, and summary files.")
    parser.add_argument("--profile", action="store_true", help="Write cProfile outputs into the artifact directory.")
    parser.add_argument("--profile-lines", default=40, type=int, help="Number of rows to include in profile_top.txt.")
    parser.add_argument("--monitor-interval", default=30.0, type=float, help="Seconds between progress heartbeat lines.")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing into a non-empty artifact directory.",
    )
    args = parser.parse_args()

    process = psutil.Process()
    ensure_writable_out_dir(args.out_dir, allow_overwrite=args.allow_overwrite)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    start_rss = process.memory_info().rss
    start_time = time.perf_counter()
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=monitor_process,
        args=(process, start_time, stop_monitor, args.monitor_interval),
        daemon=True,
    )
    monitor.start()
    trainer = BPETrainer(vocab_size=args.vocab_size, special_tokens=args.special_token)
    try:
        if args.profile:
            vocab, merges, profile_paths = train_with_profile(trainer, args.input, args.out_dir, args.profile_lines)
        else:
            vocab, merges = trainer.train(args.input)
            profile_paths = {}
    finally:
        stop_monitor.set()
        monitor.join(timeout=1)
    elapsed_seconds = time.perf_counter() - start_time
    end_rss = process.memory_info().rss

    longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))

    write_vocab(args.out_dir / "vocab.json", vocab)
    write_merges(args.out_dir / "merges.json", merges)
    write_merges_preview(args.out_dir / "merges.txt", merges)
    write_summary(
        args.out_dir / "summary.json",
        {
            "input": str(args.input),
            "vocab_size_requested": args.vocab_size,
            "vocab_size_actual": len(vocab),
            "num_merges": len(merges),
            "special_tokens": args.special_token,
            "elapsed_seconds": elapsed_seconds,
            "rss_start_bytes": start_rss,
            "rss_end_bytes": end_rss,
            "max_rss_bytes": max_rss_bytes(),
            "longest_token": token_record(longest_token, token_id=longest_id),
            "profile": profile_paths,
        },
    )

    print(f"wrote: {args.out_dir}")
    print(f"elapsed_seconds: {elapsed_seconds:.3f}")
    print(f"vocab_size_actual: {len(vocab)}")
    print(f"num_merges: {len(merges)}")
    print(f"longest_token_bytes: {len(longest_token)}")
    print(f"longest_token_text: {longest_token.decode('utf-8', errors='replace')!r}")
    if profile_paths:
        print(f"profile_top: {profile_paths['profile_top']}")


def ensure_writable_out_dir(out_dir: Path, allow_overwrite: bool) -> None:
    if allow_overwrite or not out_dir.exists():
        return
    if any(out_dir.iterdir()):
        raise SystemExit(
            f"Refusing to write into non-empty artifact directory: {out_dir}\n"
            "Choose a new --out-dir, or pass --allow-overwrite if you really want to reuse it."
        )


def monitor_process(
    process: psutil.Process,
    start_time: float,
    stop_event: threading.Event,
    interval_seconds: float,
) -> None:
    if interval_seconds <= 0:
        return

    while not stop_event.wait(interval_seconds):
        try:
            mem = process.memory_info()
            elapsed = time.perf_counter() - start_time
            print(
                "[monitor] "
                f"elapsed={format_duration(elapsed)} "
                f"rss={format_bytes(mem.rss)} "
                f"vms={format_bytes(mem.vms)} "
                f"max_rss={format_bytes(max_rss_bytes())}",
                file=sys.stderr,
                flush=True,
            )
        except psutil.Error:
            return


def train_with_profile(
    trainer: BPETrainer,
    input_path: Path,
    out_dir: Path,
    profile_lines: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], dict[str, str]]:
    profile = cProfile.Profile()
    vocab, merges = profile.runcall(trainer.train, input_path)

    stats_path = out_dir / "profile.stats"
    top_path = out_dir / "profile_top.txt"
    profile.dump_stats(stats_path)

    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(profile_lines)
    top_path.write_text(stream.getvalue(), encoding="utf-8")

    return vocab, merges, {
        "profile_stats": str(stats_path),
        "profile_top": str(top_path),
        "sort": "cumulative",
        "lines": profile_lines,
    }


def write_vocab(path: Path, vocab: dict[int, bytes]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump({str(token_id): token_record(token) for token_id, token in vocab.items()}, f, indent=2)
        f.write("\n")


def write_merges(path: Path, merges: list[tuple[bytes, bytes]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump([[token_record(left), token_record(right)] for left, right in merges], f, indent=2)
        f.write("\n")


def write_merges_preview(path: Path, merges: list[tuple[bytes, bytes]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_text = left.decode("utf-8", errors="replace")
            right_text = right.decode("utf-8", errors="replace")
            f.write(f"{left.hex()} {right.hex()}\t{left_text!r} {right_text!r}\n")


def write_summary(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def token_record(token: bytes, token_id: int | None = None) -> dict:
    record = {
        "hex": token.hex(),
        "nbytes": len(token),
        "utf8_replace": token.decode("utf-8", errors="replace"),
    }
    if token_id is not None:
        record = {"id": token_id, **record}
    return record


def format_duration(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def max_rss_bytes() -> int:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return max_rss
    return max_rss * 1024


if __name__ == "__main__":
    main()
