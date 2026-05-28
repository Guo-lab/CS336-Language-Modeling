from __future__ import annotations

import torch
import math
from einops import einsum


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax over one dimension.
    For numerical stability, subtract max(v) along `dim` before exponentiating.

    Formula:
        softmax(v)_i = exp(v_i) / sum_j exp(v_j)

    Shape:
        x: arbitrary shape
        return: same shape as x
    """
    # Keep the reduced dim as (..., 1, ...) so it can broadcast back to x.
    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    softmax_x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    return softmax_x


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply scaled dot-product attention. Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    @NOTE
    x 是 token id 被转化成向量后的 embedding，像是坐标向量，表示出 token 在 latent space 里的位置。

    Q/K/V 是从 x 这个模型的 hidden state / latent representation 线性投影出来的，
    因此，Q/K/V 里的“每个 query token 想搜索什么” “每个 key token 提供给别人匹配的标签” “每个 value token 提供什么内容”，
    是通过语言模型通过 next-token prediction，不断调整 embedding 和后面所有层的参数，训练学出来的。

    QK 的乘积是相似度，是每个 query 对每个 key 的匹配分数矩阵。
    比如，query token i 有多想看 key token j。

    对于每个 query row，有 keys 个相似度。softmax 后得到一个 keys 上的概率分布。
    每个 key 对应一个 value 向量。
    attention 输出是这些向量的加权平均，得到一个 d_v 维 hidden vector，是 query i 从所有可见 value 里混合出的新表示。

    Shape:
        Q: (..., queries, d_k)
        K: (..., keys, d_k)
        V: (..., keys, d_v)
        mask: (..., queries, keys), where True means the query may attend to the key
        return: (..., queries, d_v)
    """
    d_k = Q.shape[-1]
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    attention_scores = QK / math.sqrt(d_k)  # (..., queries, keys)

    if mask is not None:
        # Mask broadcasts over any leading batch/head dimensions
        attention_scores = attention_scores.masked_fill(mask == False, -torch.inf)
    attn_weights = softmax(attention_scores, dim=-1)  # (..., queries, keys)

    attention = einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attention
