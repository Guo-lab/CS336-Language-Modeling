from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from torch import nn

from .nn_utils import scaled_dot_product_attention


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


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct RMSNorm.
        init: gain weights initialized to 1.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Create the learnable gain parameter with shape (d_model,) and initialize it to ones.
        weight_shape = (self.d_model,)
        self.weight = nn.Parameter(torch.ones(weight_shape, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm over the final dimension (embedding_dim, aka d_model).
        Shape:
            x: (batch_size, sequence_length, d_model)
            return: (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)  # upcast to prevent overflow when we square the input

        # Normalize each activation vector over the final dimension.
        RMS = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)  # (batch, seq, 1)
        result = einsum(x / RMS, self.weight, "... d_model, d_model -> ... d_model")

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct the position-wise feed-forward network used in the Transformer block.
        """
        super().__init__()
        assert d_ff % 64 == 0, "dim of the inner feed-forward layer must be a multiple of 64"
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the position-wise feed-forward network using SwiGLU.
        Formula:
            FFN(x) = W2(SiLU(W1 x) * W3 x)
            where W1, W2, W3 denote the linear projections.
        Shape:
            x: (..., d_model)
            W1.weight, W3.weight: (d_ff, d_model)
            W2.weight: (d_model, d_ff)
            return: (..., d_model)
        """
        W1x = self.W1(x)  # (..., d_ff)
        SiLU = W1x * torch.sigmoid(W1x)  # gate, (..., d_ff)
        return self.W2(SiLU * self.W3(x))


PositionwiseFeedForward = SwiGLU


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        """
        Construct RoPE for query/key vectors and create buffers if needed.
        NOTE:
        RoPE encodes position by rotating query/key vectors instead of adding a
        position vector to token embeddings. After rotation, the attention dot
        product depends naturally on the relative offset between token positions.
        @ref
        https://learnopencv.com/rope-position-embeddings/
        https://www.youtube.com/watch?v=SMBkImDWOyQ
        NOTE:
        RoPE decomposes position into multiple frequency bands across embedding pairs.
        Low-index pairs rotate quickly and provide high-resolution local position
        information, while high-index pairs rotate slowly and preserve usable phase
        differences over longer ranges.
        Thus, different pairs give attention heads access to different distance scales,
        but semantic notions like paragraph/chapter-level relevance are learned
        by the model rather than hard-coded by RoPE.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute cos/sin tables with shape (max_seq_len, d_k / 2)
        # and store them with register_buffer(..., persistent=False).
        # q_even' = q_even * cos(angle) - q_odd  * sin(angle)
        #  q_odd' = q_odd  * cos(angle) + q_even * sin(angle)
        assert d_k % 2 == 0, "d_k must be even for RoPE pairwise rotation"

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        pair_idx = torch.arange(d_k // 2, device=device, dtype=torch.float32)

        freqs = 1 / (theta ** (2 * pair_idx / d_k))  # row vector, one frequency per pair
        angles = einsum(positions, freqs, "pos, pair -> pos pair")

        table_shape = (max_seq_len, d_k // 2)
        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)
        assert cos_table.shape == sin_table.shape == table_shape

        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE over pairs of the final dimension.
        Shape:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len),
                specifying the token positions of x along the sequence dimension.
            return: (..., seq_len, d_k)
        """
        even_q = x[..., 0::2]  # (..., seq_len, d_k//2)
        odd_q = x[..., 1::2]

        # gather-like lookup
        # cos_table stores rows for all positions up to max_seq_len.
        # token_positions selects rows from that table, preserving its batch/seq shape:
        #   (max_seq_len, d_k//2)[(..., seq_len)] -> (..., seq_len, d_k//2)
        cos = self.cos_table[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin_table[token_positions]

        out = torch.empty_like(x)
        out[..., 0::2] = even_q * cos - odd_q * sin
        out[..., 1::2] = odd_q * cos + even_q * sin
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Causal multi-head self-attention. MultiHeadSelfAttention(x) = W_o MultiHead(W_q x, W_k x, W_v x)
        Each head applies scaled dot-product attention independently.
        When RoPE is enabled, apply it to Q and K only, never V.
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.d_head = d_model // num_heads  # d_k = d_v = d_model / h

        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if rope_theta is not None:
            assert max_seq_len is not None, "max_seq_len is required when RoPE is enabled."
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Shape:
            x: (..., seq_len, d_model)
            token_positions: (..., seq_len), used only when RoPE is enabled
        """
        mask_shape = (x.shape[-2], x.shape[-2])  # (..., queries, keys)
        # True means query i may attend to key j.
        # Causal mask allows each token to attend to itself and previous tokens:
        # [[ True, False, False, ...],
        #  [ True,  True, False, ...],
        #  [ True,  True,  True, ...], ...]
        i = rearrange(torch.arange(mask_shape[0], device=x.device), "query -> query 1")
        j = rearrange(torch.arange(mask_shape[1], device=x.device), "key -> 1 key")
        masks = i >= j

        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        Q = rearrange(Q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)

        if self.rope is not None:
            assert token_positions is not None, "token_positions is required for RoPE."
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        multi_heads = scaled_dot_product_attention(Q, K, V, masks)
        multi_heads = rearrange(multi_heads, "... h seq_len d_head -> ... seq_len (h d_head)")
        return self.W_O(multi_heads)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Pre-norm Transformer block.
        Shape:
            x: (..., seq_len, d_model)
            return: (..., seq_len, d_model)
        """
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, 1e-5, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.ln2 = RMSNorm(d_model, 1e-5, device, dtype)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply the pre-norm Transformer block.
            x' = x + Causal MHA(RMSNorm(x))
            out = x' + Position-wise FFN(RMSNorm(x'))
        """
        x_hidden = self.attn(self.ln1(x), token_positions) + x
        x_out = self.ffn(self.ln2(x_hidden)) + x_hidden
        return x_out


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct a Transformer language model.
        """
        super().__init__()

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.transformer_blks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blks.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope_theta=rope_theta,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                )
            )

        self.final_ln = RMSNorm(d_model, 1e-5, device, dtype)
        self.linear_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the Transformer LM and return logits.

        Shape:
            token_ids: (..., seq_len)
            return: (..., seq_len, vocab_size)
        """
        x = self.token_embeddings(token_ids)  # (batch_size, sequence_length, embedding_dim)

        token_positions = torch.arange(x.shape[-2], device=x.device)[None, :]
        x_hidden = x
        for each_layer in self.transformer_blks:
            x_hidden = each_layer(x_hidden, token_positions)

        x_norm = self.final_ln(x_hidden)  # (batch_size, sequence_length, embedding_dim)
        vocab_logits = self.linear_head(x_norm)  # (batch_size, sequence_length, vocab_size)
        return vocab_logits
