from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch


# Generated with Codex (GPT-5).
# Toy SGD learning-rate tuning for CS336 assignment1 basics.


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                p.data -= lr / math.sqrt(t + 1) * p.grad.data
                state["t"] = t + 1
        return loss


def run_lr(lr: float, steps: int, seed: int) -> list[float]:
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    optimizer = SGD([weights], lr=lr)
    losses = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CS336 toy SGD learning-rate tuning experiment.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e1, 1e2, 1e3])
    args = parser.parse_args()

    print(f"{C.BOLD}{C.CYAN}Toy SGD learning-rate tuning{C.RESET}")
    print(f"{C.DIM}Generated with Codex (GPT-5).{C.RESET}")
    print(f"{C.DIM}loss = mean(weights ** 2), update = lr / sqrt(t + 1) * grad{C.RESET}")
    for lr in args.lrs:
        losses = run_lr(lr=lr, steps=args.steps, seed=args.seed)
        trend = "decreases" if losses[-1] < losses[0] else "diverges/increases"
        trend_color = C.GREEN if trend == "decreases" else C.RED
        print(f"\n{C.BOLD}{C.MAGENTA}lr={lr:g}{C.RESET} ({trend_color}{trend}{C.RESET})")
        for step, loss in enumerate(losses):
            print(f"  step {step:02d}: {C.YELLOW}{loss:.6g}{C.RESET}")


if __name__ == "__main__":
    main()
