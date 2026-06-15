from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def run_accumulation_case(
    name: str,
    accumulator_dtype: torch.dtype,
    value_dtype: torch.dtype,
    cast_value_to_accumulator: bool = False,
) -> None:
    s = torch.tensor(0, dtype=accumulator_dtype)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=value_dtype)
        if cast_value_to_accumulator:
            x = x.type(accumulator_dtype)
        s += x

    expected = 10.0
    error = float(s) - expected
    print(f"{name:<34} value={float(s):9.6f} dtype={s.dtype} error={error:+.6f}")


def inspect_toy_autocast_dtypes() -> None:
    if not torch.cuda.is_available():
        print("ToyModel autocast dtype inspection skipped: CUDA is not available.")
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    model = ToyModel(in_features=8, out_features=4).to(device)
    x = torch.randn(16, 8, device=device)
    targets = torch.randint(0, 4, (16,), device=device)

    activations: dict[str, torch.dtype] = {}

    def save_dtype(name: str):
        def hook(_module, _inputs, output):
            activations[name] = output.dtype

        return hook

    hooks = [
        model.fc1.register_forward_hook(save_dtype("fc1_output")),
        model.ln.register_forward_hook(save_dtype("layer_norm_output")),
        model.fc2.register_forward_hook(save_dtype("logits")),
    ]

    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, targets)
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()

    print("ToyModel autocast dtype inspection")
    print(f"{'model parameter':<24} {next(model.parameters()).dtype}")
    print(f"{'fc1 output':<24} {activations['fc1_output']}")
    print(f"{'layer norm output':<24} {activations['layer_norm_output']}")
    print(f"{'logits':<24} {activations['logits']}")
    print(f"{'loss':<24} {loss.dtype}")
    print(f"{'gradient':<24} {model.fc1.weight.grad.dtype}")


def main() -> None:
    print("Mixed-precision accumulation sanity")
    print("Expected result: 1000 * 0.01 = 10.0")
    run_accumulation_case("fp32 accumulator + fp32 value", torch.float32, torch.float32)
    run_accumulation_case("fp16 accumulator + fp16 value", torch.float16, torch.float16)
    run_accumulation_case("fp32 accumulator + fp16 value", torch.float32, torch.float16)
    run_accumulation_case(
        "fp32 accumulator + casted fp16",
        torch.float32,
        torch.float16,
        cast_value_to_accumulator=True,
    )
    print()
    inspect_toy_autocast_dtypes()


if __name__ == "__main__":
    main()
