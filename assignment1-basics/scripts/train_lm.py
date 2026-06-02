from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics import console
from cs336_basics.data import get_batch
from cs336_basics.experiment import ExperimentLogger
from cs336_basics.model import AblationConfig, TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.serialization import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import Trainer


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
    model.add_argument("--norm-mode", choices=["pre", "post", "none"], default="pre")
    model.add_argument("--position-encoding", choices=["rope", "none"], default="rope")
    model.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

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
    logging.add_argument("--tokenizer", type=Path, default=None)
    logging.add_argument("--sample-prompt", default=None)
    logging.add_argument("--sample-max-new-tokens", type=int, default=64)
    logging.add_argument("--sample-temperature", type=float, default=1.0)
    logging.add_argument("--sample-top-p", type=float, default=0.9)
    logging.add_argument("--wandb-project", default=None)
    logging.add_argument("--wandb-entity", default=None)
    logging.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)

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
            "norm_mode": args.norm_mode,
            "position_encoding": args.position_encoding,
            "ffn_type": args.ffn_type,
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
            "tokenizer": args.tokenizer,
            "sample_prompt": args.sample_prompt,
            "sample_max_new_tokens": args.sample_max_new_tokens,
            "sample_temperature": args.sample_temperature,
            "sample_top_p": args.sample_top_p,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_mode": args.wandb_mode,
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


def load_tokenizer(tokenizer_dir: Path | None) -> Tokenizer | None:
    if tokenizer_dir is None:
        return None
    return Tokenizer.from_files(
        str(tokenizer_dir / "vocab.json"),
        str(tokenizer_dir / "merges.json"),
        special_tokens=["<|endoftext|>"],
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = args_to_config(args)
    logger = ExperimentLogger(
        run_name=args.run_name,
        out_dir=args.out_dir,
        config=config,
        resume=args.resume_from is not None,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )

    console.header("Transformer LM training")
    console.info("Run directory", logger.run_dir)
    console.info("Config written to", logger.config_path)

    train_tokens = load_token_dataset("train", args.train_data, args.context_length)
    valid_tokens = load_token_dataset("valid", args.valid_data, args.context_length)
    tokenizer = load_tokenizer(args.tokenizer)
    train_summary = (
        f"tokens={len(train_tokens):,}, dtype={train_tokens.dtype}, shape={train_tokens.shape}"
    )
    valid_summary = (
        f"tokens={len(valid_tokens):,}, dtype={valid_tokens.dtype}, shape={valid_tokens.shape}"
    )
    console.info("train data", train_summary)
    console.info("valid data", valid_summary)
    if args.sample_prompt is not None and tokenizer is None:
        raise ValueError("--sample-prompt requires --tokenizer")
    if tokenizer is not None:
        console.info("tokenizer", args.tokenizer)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x, y = get_batch(
        dataset=train_tokens,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )
    console.warn(
        "batch smoke",
        f"x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}, device={x.device}",
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        ablation_config=AblationConfig(
            norm_mode=args.norm_mode,
            position_encoding=args.position_encoding,
            ffn_type=args.ffn_type,
        ),
        device=torch.device(args.device),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    start_iter = 0

    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        console.warn("resumed", f"{args.resume_from} at iteration {start_iter}")

    model.train()
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.info("model", f"parameters={num_parameters:,}, device={args.device}")
    console.info(
        "architecture",
        f"norm_mode={args.norm_mode}, position_encoding={args.position_encoding}, ffn_type={args.ffn_type}",
    )

    with torch.no_grad():
        logits = model(x)
    console.warn(
        "forward smoke",
        f"logits.shape={tuple(logits.shape)}, expected={(args.batch_size, args.context_length, args.vocab_size)}",
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_tokens=train_tokens,
        valid_tokens=valid_tokens,
        logger=logger,
        args=args,
        tokenizer=tokenizer,
        start_iter=start_iter,
    )
    try:
        trainer.run()
    finally:
        logger.close()


if __name__ == "__main__":
    main()
