from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        AdamW optimizer.
        State per parameter:
            t: update step
            m: first moment estimate
            v: second moment estimate
        """
        if lr <= 0 or eps <= 0 or weight_decay < 0:
            raise ValueError("lr, eps, and weight_decay must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr, weight_decay = group["lr"], group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                # Lazy init avoids allocating state for frozen parameters or
                # other parameters without gradients.
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                t = state["t"]
                adaptive_lr = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                p.data -= lr * weight_decay * p.data  # shrinks toward 0
                state["m"] = beta1 * state["m"] + (1 - beta1) * p.grad.data
                state["v"] = beta2 * state["v"] + (1 - beta2) * p.grad.data**2
                p.data -= adaptive_lr * state["m"] / (torch.sqrt(state["v"]) + eps)
                state["t"] = t + 1

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine learning-rate schedule with linear warmup.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_weight = (1 + math.cos(math.pi * progress)) / 2
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_weight
    else:
        return min_learning_rate


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    Clip gradients in-place so their global L2 norm is at most max_l2_norm.
    """
    global_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            global_norm_sq += p.grad.data.norm(2).item() ** 2
    global_norm = math.sqrt(global_norm_sq)
    if global_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= max_l2_norm / (global_norm + eps)
