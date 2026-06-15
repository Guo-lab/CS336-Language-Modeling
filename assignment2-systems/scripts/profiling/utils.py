from __future__ import annotations

import argparse
import statistics

import torch


def make_random_batch(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
    )
    return token_ids, targets


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_cuda_peak_memory_stats_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as error:
            if "invalid argument" not in str(error):
                raise


def print_summary(
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    times_s: list[float],
    last_loss: float,
) -> None:
    """
    Print a summary of the benchmarking results, including configuration, timing, and memory usage.

    Args:
        args: Parsed command-line configuration for this benchmark run.
        device: Torch device used for model execution and timing.
        model: Model being benchmarked; used here to report parameter count.

        token_ids: Random input token IDs, shaped ``(batch_size, context_length)``.
        targets: Random target token IDs, shaped ``(batch_size, context_length)``.
        d_model: Effective model hidden size after preset/default resolution.
        d_ff: Effective feed-forward hidden size after preset/default resolution.
        num_layers: Effective number of Transformer layers.
        num_heads: Effective number of attention heads.

        times_s: Timed step durations in seconds; warmup steps are excluded.
        last_loss: Loss from the final measured step. This is a workload sanity
            check, not a training-quality metric.

    Printed fields:
        mean_ms_per_step: Mean of ``times_s`` converted to milliseconds.
        std_ms_per_step: Sample standard deviation of ``times_s`` converted to
            milliseconds; zero when there is only one measured step.

        tokens_per_step: ``batch_size * context_length``.
        tokens_per_second: ``tokens_per_step / mean_step_time``. In backward or
            optimizer modes, this still counts input tokens processed per full
            benchmark step, not generated tokens.

        peak_memory_allocated_gib: CUDA-only PyTorch live tensor memory peak
            since ``torch.cuda.reset_peak_memory_stats`` was last called.
        peak_memory_reserved_gib: CUDA-only PyTorch caching allocator reserved
            memory peak over the same interval.
    """
    mean_s = statistics.fmean(times_s)
    std_s = statistics.stdev(times_s) if len(times_s) > 1 else 0.0
    tokens_per_step = args.batch_size * args.context_length
    tokens_per_second = tokens_per_step / mean_s
    num_params = sum(p.numel() for p in model.parameters())

    print(f"model_size={args.model_size}")
    print(f"mode={args.mode}")
    print(f"device={device}")
    print(f"dtype={args.dtype}")
    print(f"d_model={d_model}")
    print(f"d_ff={d_ff}")
    print(f"num_layers={num_layers}")
    print(f"num_heads={num_heads}")
    print(f"num_params={num_params:,}")
    print(f"token_ids.shape={tuple(token_ids.shape)}")
    print(f"targets.shape={tuple(targets.shape)}")
    print(f"warmup_steps={args.warmup_steps}")
    print(f"steps={args.steps}")
    print(f"lr={args.lr}")
    print(f"weight_decay={args.weight_decay}")
    print(f"mixed_precision={args.mixed_precision}")
    print(f"autocast_dtype={args.autocast_dtype}")
    print(f"nvtx={args.nvtx}")
    print(f"memory_profile={args.memory_profile}")
    print(f"memory_snapshot_path={args.memory_snapshot_path}")
    print(f"memory_history_max_entries={args.memory_history_max_entries}")
    print(f"last_loss={last_loss:.6f}")
    print(f"mean_ms_per_step={mean_s * 1_000:.3f}")
    print(f"std_ms_per_step={std_s * 1_000:.3f}")
    print(f"tokens_per_step={tokens_per_step:,}")
    print(f"tokens_per_second={tokens_per_second:,.2f}")

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        print(f"peak_memory_allocated_gib={peak_allocated / 1024**3:.3f}")
        print(f"peak_memory_reserved_gib={peak_reserved / 1024**3:.3f}")
