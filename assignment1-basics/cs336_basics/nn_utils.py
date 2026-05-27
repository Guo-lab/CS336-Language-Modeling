from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax over one dimension.
    Shape:
        x: arbitrary shape
        return: same shape as x
    """
    raise NotImplementedError
