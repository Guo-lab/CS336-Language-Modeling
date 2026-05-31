from __future__ import annotations

import torch

from .nn_utils import softmax


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> torch.Tensor:
    """
    Sample one token ID from next-token logits.

    Recall:
        temperature rescales logits before softmax.
        top-p keeps the smallest high-probability set whose cumulative mass reaches p.

    Shape:
        logits: (..., vocab_size)
        return: (..., 1), sampled token IDs ready to append
    """
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, dim=-1)  # (..., vocab_size)

    def sample_index_from_probs(probs: torch.Tensor) -> torch.Tensor:
        """
        Input: probabilities tensor of shape (..., vocab_size)
        Output: sampled indices of shape (..., 1)
        """
        vocab_size = probs.shape[-1]
        flat_probs = probs.reshape(-1, vocab_size)
        sample = torch.multinomial(flat_probs, num_samples=1)
        return sample.reshape(*probs.shape[:-1], 1)

    if top_p is not None:
        sorted_probs, sorted_token_ids = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        keep = cumulative_probs <= top_p  # (..., vocab_size)
        keep[..., 1:] = keep[..., :-1].clone()  # right shift the mask to keep
        keep[..., 0] = True
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        rank = sample_index_from_probs(sorted_probs)  # (..., 1)
        return sorted_token_ids.gather(dim=-1, index=rank)

    return sample_index_from_probs(probs)


def decode(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    context_length: int | None = None,
) -> torch.Tensor:
    """
    Generate tokens autoregressively from a prompt.

    Recall:
        model(tokens) returns logits for every position.
        The last position logits predict the next token.
        The sampled token is appended and used as input for the next step.

    Shape:
        prompt_token_ids: (..., seq_len)
        return: (..., seq_len + generated_len)
    """
    generated_len = 0
    while generated_len < max_new_tokens:
        input_ids = prompt_token_ids
        if context_length is not None:
            input_ids = prompt_token_ids[..., -context_length:]
        with torch.no_grad():
            logits = model(input_ids)  # (..., seq_len, vocab_size)

        next_token_logits = logits[..., -1, :]
        next_token = sample_next_token(next_token_logits, temperature=temperature, top_p=top_p)

        prompt_token_ids = torch.cat([prompt_token_ids, next_token], dim=-1)
        generated_len += 1
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return prompt_token_ids
