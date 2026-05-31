from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language-model input sequences and next-token targets.
    Shape:
        dataset: (num_tokens,)
        return x: (batch_size, context_length)
        return y: (batch_size, context_length)
    """
    sample_range_end = len(dataset) - context_length
    sample_start_idx = np.random.randint(0, sample_range_end, batch_size)  # (batch_size,)
    offsets = np.arange(context_length)  # (context_length,)

    # Each row starts at one sampled index and spans context_length consecutive tokens.
    x_idx = sample_start_idx[:, None] + offsets[None, :]  # (batch_size, context_length)
    x = torch.from_numpy(dataset[x_idx]).to(device)
    y_idx = sample_start_idx[:, None] + offsets[None, :] + 1
    y = torch.from_numpy(dataset[y_idx]).to(device)
    return x, y
