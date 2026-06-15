from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from config import DTYPES


@contextmanager
def maybe_nvtx_range(name: str, enabled: bool) -> Iterator[None]:
    context = (
        torch.cuda.nvtx.range(name) if enabled and torch.cuda.is_available() else nullcontext()
    )
    with context:
        yield


def autocast_context(args: argparse.Namespace, device: torch.device):
    return (
        torch.autocast(device_type=device.type, dtype=DTYPES[args.autocast_dtype])
        if args.mixed_precision and device.type in {"cuda", "cpu"}
        else nullcontext()
    )


def cuda_memory_profile_enabled(args: argparse.Namespace, device: torch.device) -> bool:
    return args.memory_profile and device.type == "cuda"


@contextmanager
def memory_history_context(
    args: argparse.Namespace, device: torch.device
) -> Iterator[None]:
    if not cuda_memory_profile_enabled(args, device):
        yield
        return

    torch.cuda.memory._record_memory_history(
        max_entries=args.memory_history_max_entries
    )
    try:
        yield
        Path(args.memory_snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(args.memory_snapshot_path)
        print(f"memory_snapshot_written={args.memory_snapshot_path}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
