from __future__ import annotations

import math
from dataclasses import dataclass


# Generated with Codex (GPT-5).
# AdamW training memory/FLOPs accounting for CS336 assignment1 basics.


BYTES_PER_FLOAT32 = 4
H100_TF32_FLOPS = 495e12
MFU = 0.50


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"


@dataclass(frozen=True)
class Config:
    name: str
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int

    @property
    def d_ff(self) -> float:
        return 8 * self.d_model / 3


def parameter_count(c: Config) -> float:
    v, l, d, f = c.vocab_size, c.num_layers, c.d_model, c.d_ff
    return 2 * v * d + l * (4 * d * d + 3 * d * f + 2 * d) + d


def activation_count_per_batch_item(c: Config) -> float:
    return sum(activation_breakdown_per_batch_item(c).values())


def activation_breakdown_per_batch_item(c: Config) -> dict[str, float]:
    t, l, d, h, f, v = (
        c.context_length,
        c.num_layers,
        c.d_model,
        c.num_heads,
        c.d_ff,
        c.vocab_size,
    )
    return {
        "block RMSNorms": l * 2 * t * d,
        "MHA QKV projections": l * 3 * t * d,
        "MHA QK^T scores": l * h * t * t,
        "MHA softmax": l * h * t * t,
        "MHA weighted values": l * t * d,
        "MHA output projection": l * t * d,
        "SwiGLU W1/W3": l * 2 * t * f,
        "SwiGLU SiLU gate": l * t * f,
        "SwiGLU elementwise product": l * t * f,
        "SwiGLU W2": l * t * d,
        "final RMSNorm": t * d,
        "output embedding logits": t * v,
        "cross-entropy on logits": t * v,
    }


def forward_flops_per_batch_item(c: Config) -> float:
    t, l, d, f, v = c.context_length, c.num_layers, c.d_model, c.d_ff, c.vocab_size
    return (
        l * 3 * matmul_flops(t, d, d)
        + l * 2 * t * t * d
        + l * 2 * t * t * d
        + l * matmul_flops(t, d, d)
        + l * 3 * matmul_flops(t, d, f)
        + matmul_flops(t, d, v)
    )


def matmul_flops(m: float, n: float, p: float) -> float:
    return 2 * m * n * p


def adamw_flops(parameter_count_value: float) -> float:
    # Per parameter, roughly:
    # weight decay: mul + sub
    # m update: 2 mul + add
    # v update: square + 2 mul + add
    # Adam update: sqrt + add + div + mul + sub
    return 14 * parameter_count_value


def gib(n_bytes: float) -> float:
    return n_bytes / 1024**3


def human(n: float) -> str:
    units = ["", "K", "M", "B", "T", "P", "E"]
    x = float(n)
    for unit in units:
        if abs(x) < 1000:
            return f"{x:,.2f}{unit}"
        x /= 1000
    return f"{x:,.2f}Z"


def formula(label: str, expr: str, note: str = "") -> None:
    suffix = f"  {C.DIM}# {note}{C.RESET}" if note else ""
    print(f"  {label:<24} {C.YELLOW}{expr}{C.RESET}{suffix}")


def print_formula_notes() -> None:
    print(f"{C.BOLD}{C.CYAN}AdamW training resource accounting{C.RESET}")
    print(f"{C.DIM}Generated with Codex (GPT-5). Assume float32 for every tensor.{C.RESET}")
    print(f"\n{C.BOLD}{C.MAGENTA}Algebraic setup{C.RESET}")
    print(f"  {C.DIM}B=batch_size, V=vocab_size, T=context_length, L=num_layers, D=d_model, H=num_heads, F=8D/3{C.RESET}")
    formula("P", "2VD + L(4D^2 + 3DF + 2D) + D", "trainable parameters")
    print(f"  {C.BOLD}block_activations{C.RESET}")
    formula("  RMSNorms", "2TD", "ln1 and ln2")
    formula("  QKV projections", "3TD", "Q, K, V")
    formula("  QK^T scores", "HT^2", "attention scores")
    formula("  softmax", "HT^2", "attention probabilities")
    formula("  weighted values", "TD", "attention output before W_O")
    formula("  output projection", "TD", "W_O output")
    formula("  SwiGLU W1/W3", "2TF", "two up-projection branches")
    formula("  SiLU gate", "TF", "activation on gate branch")
    formula("  elementwise product", "TF", "SwiGLU gated hidden")
    formula("  SwiGLU W2", "TD", "down-projection output")
    formula("block_activations", "8TD + 4TF + 2HT^2", "simplified")
    print(f"  {C.BOLD}activations_per_batch_item{C.RESET}")
    formula("  transformer blocks", "L * block_activations")
    formula("  final RMSNorm", "TD")
    formula("  output logits", "TV")
    formula("  cross-entropy", "TV")
    formula("activations_per_batch_item", "L*block_activations + TD + TV + TV")
    formula("", "= L(8TD + 4TF + 2HT^2) + TD + 2TV", "simplified")
    formula("memory_params", "4 * P bytes", "parameters")
    formula("memory_gradients", "4 * P bytes", "gradients")
    formula("memory_optim_state", "8 * P bytes", "AdamW m and v")
    formula("memory_activations", "4 * B * activations_per_batch_item bytes", "saved activations")
    formula("total_peak_memory", "16 * P + 4 * B * activations_per_batch_item bytes", "params + grads + state + activations")


def print_xl_memory(c: Config) -> None:
    p = parameter_count(c)
    activation_parts = activation_breakdown_per_batch_item(c)
    a_floats = activation_count_per_batch_item(c)
    a_bytes = BYTES_PER_FLOAT32 * a_floats
    b_bytes = 16 * p
    limit_bytes = 80 * 1024**3
    max_batch = math.floor((limit_bytes - b_bytes) / a_bytes)

    print(f"\n{C.BOLD}{C.MAGENTA}{c.name} memory{C.RESET}")
    print(
        f"  {C.DIM}V={c.vocab_size}, T={c.context_length}, L={c.num_layers}, "
        f"D={c.d_model}, H={c.num_heads}, F=8D/3={c.d_ff:.2f}{C.RESET}"
    )
    print(f"  parameters P:             {C.GREEN}{human(p)}{C.RESET}")
    print(f"  activations per batch:    {C.GREEN}{human(a_floats)} floats{C.RESET}")
    print(f"  total memory expression:  {C.YELLOW}{gib(a_bytes):.2f} GiB * batch_size + {gib(b_bytes):.2f} GiB{C.RESET}")
    print(f"  max batch under 80 GiB:   {C.YELLOW}{max_batch}{C.RESET}")
    print(f"\n{C.BOLD}{C.CYAN}Activation breakdown per batch item{C.RESET}")
    for name, value in activation_parts.items():
        print(f"  {name:<30} {C.GREEN}{human(value):>10} floats{C.RESET}  {C.DIM}{value / a_floats:6.2%}{C.RESET}")


def print_flops_and_time(c: Config, batch_size: int, steps: int) -> None:
    p = parameter_count(c)
    fwd = forward_flops_per_batch_item(c)
    opt = adamw_flops(p)
    train_step = batch_size * 3 * fwd + opt
    throughput = H100_TF32_FLOPS * MFU
    seconds = steps * train_step / throughput

    print(f"\n{C.BOLD}{C.MAGENTA}AdamW compute and H100 time{C.RESET}")
    print(f"  {C.BOLD}AdamW FLOPs per parameter{C.RESET}")
    print(f"    weight decay     {C.YELLOW}mul + sub = 2{C.RESET}  {C.DIM}# theta <- theta - lr * wd * theta{C.RESET}")
    print(f"    m update         {C.YELLOW}2 mul + add = 3{C.RESET}  {C.DIM}# m <- beta1*m + (1-beta1)*g{C.RESET}")
    print(f"    v update         {C.YELLOW}square + 2 mul + add = 4{C.RESET}  {C.DIM}# v <- beta2*v + (1-beta2)*g^2{C.RESET}")
    print(f"    Adam update      {C.YELLOW}sqrt + add + div + mul + sub = 5{C.RESET}  {C.DIM}# theta update with m/sqrt(v){C.RESET}")
    print(f"  AdamW optimizer FLOPs:     {C.GREEN}~14 FLOPs/parameter = {human(opt)} FLOPs per optimizer step{C.RESET}")
    print(f"  forward FLOPs / sequence: {C.GREEN}{human(fwd)}{C.RESET}")
    print(f"  {C.DIM}training step uses forward + backward, and backward is assumed to be 2x forward.{C.RESET}")
    print(f"  {C.DIM}with batch_size={batch_size}: {batch_size} * 3 * {human(fwd)} + {human(opt)}{C.RESET}")
    print(
        f"  train step FLOPs:         {C.GREEN}batch_size * 3 * forward + AdamW "
        f"= {train_step / 1e12:,.2f}T ({human(train_step)}){C.RESET}"
    )
    print(f"  H100 effective FLOPs/s:   {C.GREEN}{human(throughput)} FLOP/s at 50% MFU{C.RESET}")
    print(f"  {steps:,} steps, batch={batch_size}: {C.YELLOW}{seconds / 3600:.2f} hours{C.RESET}")


def main() -> None:
    gpt2_xl = Config(
        name="GPT-2 XL shaped",
        vocab_size=50_257,
        context_length=1_024,
        num_layers=48,
        d_model=1_600,
        num_heads=25,
    )
    print_formula_notes()
    print_xl_memory(gpt2_xl)
    print_flops_and_time(gpt2_xl, batch_size=1024, steps=400_000)


if __name__ == "__main__":
    main()
