from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from cs336_basics.experiment import ExperimentLogger


class TerminalStyle:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")

    data = parser.add_argument_group("data")
    data.add_argument("--train-data", type=Path, required=True)
    data.add_argument("--valid-data", type=Path, required=True)
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

    print(f"{TerminalStyle.BOLD}{TerminalStyle.CYAN}Transformer LM training{TerminalStyle.RESET}")
    print(f"{TerminalStyle.GREEN}Run directory:{TerminalStyle.RESET} {logger.run_dir}")
    print(f"{TerminalStyle.GREEN}Config written to:{TerminalStyle.RESET} {logger.config_path}")
    print(f"{TerminalStyle.DIM}Training loop is not implemented yet.{TerminalStyle.RESET}")
    raise NotImplementedError("Training loop is not implemented yet.")


if __name__ == "__main__":
    main()
