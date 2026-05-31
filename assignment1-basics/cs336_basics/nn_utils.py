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


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute average cross-entropy loss from logits and target class IDs.
    模型每个位置都预测 “下一个 token”，cross entropy 惩罚模型给真正下一个 token 的概率不够高

    p(x_{i+1} | x_1:i) = softmax(o_i)["<target i-th token>"]
                       = exp(o_i[x_{i+1}]) / sum_a exp(o_i[a])
    Formula:
        loss_i = -log softmax(logits_i)[target_i]
        where logits at the i-th position area vector of unnormalized scores over the vocabulary,
        shape (vocab_size,).

    -log p(x_{i+1} | x_1:i) = -log(exp(o_i[x_{i+1}])) + log(sum_a exp(o_i[a]))
                            = -o_i[target_i] + logsumexp(o_i)

    Shape:
        logits: (..., vocab_size)
        targets: (...), target_i is the next token ID x_{i+1}
        return: scalar average loss
    """
    assert logits.shape[:-1] == targets.shape, "logits and targets must share batch-like dims"
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)  # (..., vocab_size) -> (N, vocab_size)
    flat_targets = targets.reshape(-1)  # (...) -> (N,)
    rows = torch.arange(flat_logits.size(0), device=flat_logits.device)
    target_token_score = flat_logits[rows, flat_targets]

    max_logits = torch.max(flat_logits, dim=-1).values  # (N,)
    shifted_logits = flat_logits - max_logits.unsqueeze(-1)
    # (N, vocab_size) -> (N,)
    vocab_tokens_score = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    vocab_tokens_score = vocab_tokens_score + max_logits

    cross_entropy_loss = -target_token_score + vocab_tokens_score
    return cross_entropy_loss.mean()
