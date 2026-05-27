from __future__ import annotations

import math

import torch
from einops import einsum
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct a linear transformation module.
        init: N(0, 2 / (d_in + d_out)) truncated to [-3 std, 3 std].
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight_shape = (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(weight_shape, device=device, dtype=dtype))

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        Shape:
            x: (..., in_features)
            return: (..., out_features)
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct an embedding lookup table.
        init: N(0, 1) truncated to [-3, 3].
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # TODO: create self.weight as an nn.Parameter with shape
        # (num_embeddings, embedding_dim), then initialize it with trunc_normal_.

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embedding vectors for token IDs.
        Shape:
            token_ids: (...)
            return: (..., embedding_dim)
        """
        # TODO: return rows from self.weight indexed by token_ids.
        raise NotImplementedError
