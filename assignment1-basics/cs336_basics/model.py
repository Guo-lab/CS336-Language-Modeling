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
        Construct an embedding lookup table. Each row stores the embedding of one token.
        init: N(0, 1) truncated to [-3, 3].
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim  # Dimension of the embedding vectors, d_model

        weight_shape = (num_embeddings, embedding_dim)  # (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty(weight_shape, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embedding vectors for token IDs.
        Shape:
            token_ids: (...), usually (batch_size, sequence_length) given by the tokenizer
                        Each position contains one token ID in [0, num_embeddings).
            return: (..., embedding_dim), usually (batch_size, sequence_length, embedding_dim)
        """
        return self.weight[token_ids]
