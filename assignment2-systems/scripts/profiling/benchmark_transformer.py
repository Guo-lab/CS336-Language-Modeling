from __future__ import annotations

import argparse

import torch

from cs336_basics.model import TransformerLM

MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up a TransformerLM benchmarking run with random data."
    )
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZES), default="small")
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "full_w_optimizer_step"],
        default="forward",
        help="Step type to benchmark.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Vocabulary size.")
    parser.add_argument("--context-length", type=int, default=512, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--d-model", type=int, default=None, help="Override preset d_model.")
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10_000.0, help="RoPE theta.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Untimed warmup steps.")
    parser.add_argument("--steps", type=int, default=10, help="Timed measurement steps.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer's.")
    parser.add_argument("--mixed-precision", action="store_true", help="Use autocast.")
    parser.add_argument(
        "--autocast-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Autocast dtype.",
    )
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges.")
    parser.add_argument("--memory-profile", action="store_true", help="Record mem snapshot.")
    parser.add_argument(
        "--memory-snapshot-path",
        default="memory_snapshot.pickle",
        help="Memory snapshot path.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
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
    print("timer_status=not implemented")


if __name__ == "__main__":
    main()
