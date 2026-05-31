from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.experiment import ExperimentLogger
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW, clip_gradients, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint


class TerminalStyle:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    data = parser.add_argument_group("data")
    data.add_argument("--train-data", type=Path, required=True, help="Training token IDs.")
    data.add_argument("--valid-data", type=Path, required=True, help="Validation token IDs.")
    data.add_argument("--device", default="cpu")
    data.add_argument("--seed", type=int, default=0)

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, required=True)
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-heads", type=int, default=8)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10_000.0)

    optimizer = parser.add_argument_group("optimizer")
    optimizer.add_argument("--batch-size", type=int, default=32)
    optimizer.add_argument("--max-iters", type=int, default=10_000)
    optimizer.add_argument("--max-lr", type=float, default=3e-4)
    optimizer.add_argument("--min-lr", type=float, default=3e-5)
    optimizer.add_argument("--warmup-iters", type=int, default=1_000)
    optimizer.add_argument("--cosine-cycle-iters", type=int, default=10_000)
    optimizer.add_argument("--weight-decay", type=float, default=0.1)
    optimizer.add_argument("--beta1", type=float, default=0.9)
    optimizer.add_argument("--beta2", type=float, default=0.95)
    optimizer.add_argument("--eps", type=float, default=1e-8)
    optimizer.add_argument("--grad-clip", type=float, default=1.0)

    logging = parser.add_argument_group("logging")
    logging.add_argument("--run-name", required=True)
    logging.add_argument("--out-dir", type=Path, default=Path("artifacts/lm_experiments"))
    logging.add_argument("--log-every", type=int, default=10)
    logging.add_argument("--eval-every", type=int, default=100)
    logging.add_argument("--eval-iters", type=int, default=20)
    logging.add_argument("--sample-every", type=int, default=500)

    checkpoint = parser.add_argument_group("checkpoint")
    checkpoint.add_argument("--save-every", type=int, default=1_000)
    checkpoint.add_argument("--resume-from", type=Path, default=None)

    return parser


def args_to_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data": {
            "train_data": args.train_data,
            "valid_data": args.valid_data,
            "device": args.device,
            "seed": args.seed,
        },
        "model": {
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "num_layers": args.num_layers,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
        },
        "optimizer": {
            "batch_size": args.batch_size,
            "max_iters": args.max_iters,
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup_iters": args.warmup_iters,
            "cosine_cycle_iters": args.cosine_cycle_iters,
            "weight_decay": args.weight_decay,
            "betas": [args.beta1, args.beta2],
            "eps": args.eps,
            "grad_clip": args.grad_clip,
        },
        "logging": {
            "run_name": args.run_name,
            "out_dir": args.out_dir,
            "log_every": args.log_every,
            "eval_every": args.eval_every,
            "eval_iters": args.eval_iters,
            "sample_every": args.sample_every,
        },
        "checkpoint": {
            "save_every": args.save_every,
            "resume_from": args.resume_from,
        },
    }


def load_token_dataset(name: str, path: Path, context_length: int) -> np.ndarray:
    dataset = np.load(path, mmap_mode="r")
    if dataset.ndim != 1:
        raise ValueError(f"Expected a 1D token array, got shape {dataset.shape} from {path}")
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError(f"Expected integer token IDs, got dtype {dataset.dtype} from {path}")
    if len(dataset) <= context_length:
        raise ValueError(
            f"{name} dataset has {len(dataset)} tokens, "
            f"but context_length={context_length} needs at least {context_length + 1}"
        )
    return dataset


def print_dataset_summary(name: str, dataset: np.ndarray) -> None:
    print(
        f"{TerminalStyle.GREEN}{name}:{TerminalStyle.RESET} "
        f"tokens={len(dataset):,}, dtype={dataset.dtype}, shape={dataset.shape}"
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(args: argparse.Namespace) -> TransformerLM:
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
    )


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    args: argparse.Namespace,
) -> float:
    was_training = model.training
    model.eval()

    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(
            dataset=dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )
        logits = model(x)
        losses.append(cross_entropy(logits, y).item())

    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = args_to_config(args)
    logger = ExperimentLogger(
        run_name=args.run_name,
        out_dir=args.out_dir,
        config=config,
        resume=args.resume_from is not None,
    )

    print(
        f"{TerminalStyle.BOLD}{TerminalStyle.CYAN}Transformer LM training{TerminalStyle.RESET}"
    )
    print(f"{TerminalStyle.GREEN}Run directory:{TerminalStyle.RESET} {logger.run_dir}")
    print(f"{TerminalStyle.GREEN}Config written to:{TerminalStyle.RESET} {logger.config_path}")

    train_tokens = load_token_dataset("train", args.train_data, args.context_length)
    valid_tokens = load_token_dataset("valid", args.valid_data, args.context_length)
    print_dataset_summary("train data", train_tokens)
    print_dataset_summary("valid data", valid_tokens)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x, y = get_batch(
        dataset=train_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )
    print(
        f"{TerminalStyle.YELLOW}batch smoke:{TerminalStyle.RESET} "
        f"x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}, device={x.device}"
    )

    model = build_model(args)
    optimizer = build_optimizer(args, model)
    start_iter = 0

    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(
            f"{TerminalStyle.YELLOW}resumed:{TerminalStyle.RESET} "
            f"{args.resume_from} at iteration {start_iter}"
        )

    model.train()
    num_parameters = count_parameters(model)
    print(
        f"{TerminalStyle.GREEN}model:{TerminalStyle.RESET} "
        f"parameters={num_parameters:,}, device={args.device}"
    )

    with torch.no_grad():
        logits = model(x)
    print(
        f"{TerminalStyle.YELLOW}forward smoke:{TerminalStyle.RESET} "
        f"logits.shape={tuple(logits.shape)}, expected={(args.batch_size, args.context_length, args.vocab_size)}"
    )

    print(
        f"{TerminalStyle.BOLD}{TerminalStyle.CYAN}Starting training loop{TerminalStyle.RESET} "
        f"{TerminalStyle.DIM}from iteration {start_iter} to {args.max_iters}{TerminalStyle.RESET}"
    )

    for step in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        set_learning_rate(optimizer, lr)

        x, y = get_batch(
            dataset=train_tokens,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            clip_gradients(model.parameters(), args.grad_clip)
        optimizer.step()

        iteration = step + 1
        if iteration % args.log_every == 0 or iteration == 1:
            train_loss = loss.item()
            logger.log_metrics(iteration, {"train_loss": train_loss, "lr": lr})
            print(
                f"{TerminalStyle.GREEN}step {iteration:>6}:{TerminalStyle.RESET} "
                f"train_loss={train_loss:.4f}, lr={lr:.6g}"
            )

        if iteration % args.eval_every == 0 or iteration == args.max_iters:
            valid_loss = estimate_loss(model, valid_tokens, args)
            logger.log_metrics(iteration, {"valid_loss": valid_loss})
            print(
                f"{TerminalStyle.YELLOW}eval {iteration:>6}:{TerminalStyle.RESET} "
                f"valid_loss={valid_loss:.4f}"
            )

        if iteration % args.save_every == 0 or iteration == args.max_iters:
            checkpoint_path = logger.checkpoint_path(iteration)
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(
                f"{TerminalStyle.DIM}checkpoint saved: {checkpoint_path}"
                f"{TerminalStyle.RESET}"
            )


if __name__ == "__main__":
    main()
