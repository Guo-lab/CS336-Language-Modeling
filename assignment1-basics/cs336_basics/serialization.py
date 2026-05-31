from __future__ import annotations

import os
from pathlib import Path
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model state, optimizer state, and training iteration.

    Recall:
        model.state_dict() stores model parameters and buffers.
        optimizer.state_dict() stores optimizer hyperparameters and per-parameter state.
        torch.save(...) can write either to a path or a file-like object.
    """
    obj: dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Restore model and optimizer state from a checkpoint and return the iteration.

    Recall:
        torch.load(...) recovers the object saved by torch.save(...).
        model.load_state_dict(...) and optimizer.load_state_dict(...) restore state.
    """
    obj: dict = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]


def to_jsonable(value):
    """Convert common config/metric values into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value
