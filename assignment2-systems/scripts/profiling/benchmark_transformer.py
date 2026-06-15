from __future__ import annotations

import argparse
import timeit
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F


from cs336_basics.model import TransformerLM

from config import DTYPES, MODEL_SIZES, parse_args
from utils import make_random_batch, print_summary, synchronize_if_needed


@contextmanager
def maybe_nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def autocast_context(args: argparse.Namespace, device: torch.device):
    if not args.mixed_precision:
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=DTYPES[args.autocast_dtype],
    )


def forward_loss(
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    with maybe_nvtx_range("forward", args.nvtx):
        with autocast_context(args, device):
            logits = model(token_ids)

    with maybe_nvtx_range("loss", args.nvtx):
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )


def run_step(
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    if args.mode == "forward":
        model.eval()
        with torch.no_grad():
            loss = forward_loss(args, device, model, token_ids, targets)
        return float(loss.detach())

    assert optimizer is not None
    model.train()

    with maybe_nvtx_range("zero_grad", args.nvtx):
        optimizer.zero_grad(set_to_none=True)

    loss = forward_loss(args, device, model, token_ids, targets)

    with maybe_nvtx_range("backward", args.nvtx):
        loss.backward()

    if args.mode == "full_w_optimizer_step":
        with maybe_nvtx_range("optimizer_step", args.nvtx):
            optimizer.step()

    return float(loss.detach())


def time_steps(
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[list[float], float]:
    last_loss = float("nan")

    for _ in range(args.warmup_steps):
        last_loss = run_step(args, device, model, optimizer, token_ids, targets)
    synchronize_if_needed(device)

    times_s: list[float] = []
    for _ in range(args.steps):
        synchronize_if_needed(device)
        start = timeit.default_timer()
        last_loss = run_step(args, device, model, optimizer, token_ids, targets)
        synchronize_if_needed(device)
        times_s.append(timeit.default_timer() - start)

    return times_s, last_loss


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    token_ids, targets = make_random_batch(args, device)

    size = MODEL_SIZES[args.model_size]
    d_model = args.d_model if args.d_model is not None else size["d_model"]
    d_ff = args.d_ff if args.d_ff is not None else size["d_ff"]
    num_layers = args.num_layers if args.num_layers is not None else size["num_layers"]
    num_heads = args.num_heads if args.num_heads is not None else size["num_heads"]

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=DTYPES[args.dtype],
    )

    optimizer = None
    if args.mode != "forward":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times_s, last_loss = time_steps(args, device, model, optimizer, token_ids, targets)

        print_summary(
            args=args,
            device=device,
            model=model,
            token_ids=token_ids,
            targets=targets,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            times_s=times_s,
            last_loss=last_loss,
        )
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
