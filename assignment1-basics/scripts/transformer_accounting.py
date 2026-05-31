from __future__ import annotations

from dataclasses import dataclass


# Generated with Codex (GPT-5).
# Transformer LM resource accounting for CS336 assignment1 basics.


VOCAB_SIZE = 50_257
DEFAULT_CONTEXT_LENGTH = 1_024
BYTES_PER_FP32 = 4


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
    num_layers: int
    d_model: int
    num_heads: int
    context_length: int = DEFAULT_CONTEXT_LENGTH
    vocab_size: int = VOCAB_SIZE

    @property
    def d_ff(self) -> int:
        return round_to_multiple(8 * self.d_model / 3, 64)

    @property
    def d_head(self) -> int:
        assert self.d_model % self.num_heads == 0
        return self.d_model // self.num_heads


def round_to_multiple(value: float, multiple: int) -> int:
    return int(round(value / multiple) * multiple)


def params(config: Config) -> dict[str, int]:
    embedding = config.vocab_size * config.d_model
    per_layer_attention = 4 * config.d_model * config.d_model
    per_layer_ffn = 3 * config.d_model * config.d_ff
    per_layer_rmsnorm = 2 * config.d_model
    layers = config.num_layers * (per_layer_attention + per_layer_ffn + per_layer_rmsnorm)
    final_rmsnorm = config.d_model
    lm_head = config.vocab_size * config.d_model
    return {
        "token embeddings": embedding,
        "transformer layers": layers,
        "final RMSNorm": final_rmsnorm,
        "LM head": lm_head,
    }


def flops(config: Config) -> dict[str, int]:
    t = config.context_length
    d = config.d_model
    d_ff = config.d_ff
    l = config.num_layers
    return {
        "QKV projections": l * 3 * matmul_flops(t, d, d),
        "attention scores QK^T": l * 2 * t * t * d,
        "attention weighted values": l * 2 * t * t * d,
        "attention output projection": l * matmul_flops(t, d, d),
        "SwiGLU W1/W3/W2": l * 3 * matmul_flops(t, d, d_ff),
        "LM head logits": matmul_flops(t, d, config.vocab_size),
    }


def matmul_flops(m: int, n: int, p: int) -> int:
    return 2 * m * n * p


def human_count(n: int) -> str:
    units = ["", "K", "M", "B", "T", "P"]
    x = float(n)
    for unit in units:
        if abs(x) < 1000:
            return f"{x:,.2f}{unit}"
        x /= 1000
    return f"{x:,.2f}E"


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for unit in units:
        if abs(x) < 1024:
            return f"{x:,.2f} {unit}"
        x /= 1024
    return f"{x:,.2f} PiB"


def print_table(title: str, rows: list[tuple[str, str, str]]) -> None:
    name_width = max(len(row[0]) for row in rows)
    value_width = max(len(row[1]) for row in rows)
    print(f"\n{C.BOLD}{C.CYAN}{title}{C.RESET}")
    for name, value, extra in rows:
        print(f"  {name:<{name_width}}  {C.GREEN}{value:>{value_width}}{C.RESET}  {C.DIM}{extra}{C.RESET}")


def formula(label: str, expr: str, note: str = "") -> None:
    suffix = f"  {C.DIM}# {note}{C.RESET}" if note else ""
    print(f"    {label:<28} {C.YELLOW}{expr}{C.RESET}{suffix}")


def print_formula_map() -> None:
    print(f"\n{C.BOLD}{C.CYAN}Formula map{C.RESET}")
    print(f"  {C.DIM}Symbols: L=num_layers, T=context_length, D=d_model, F=d_ff, V=vocab_size, H=num_heads{C.RESET}")
    print(f"  {C.BOLD}Parameters{C.RESET}")
    formula("token embeddings:", "V * D")
    formula("per-layer attention:", "4 * D * D", "W_Q, W_K, W_V, W_O")
    formula("per-layer SwiGLU FFN:", "3 * D * F", "W1, W3, W2")
    formula("per-layer RMSNorms:", "2 * D", "ln1, ln2")
    formula("transformer layers:", "L * (4D^2 + 3DF + 2D)")
    formula("final RMSNorm:", "D")
    formula("LM head:", "V * D")
    print(f"  {C.BOLD}Forward FLOPs{C.RESET}")
    formula("QKV projections:", "L * 3 * 2TDD")
    formula("attention scores QK^T:", "L * 2TTD", "all heads combined")
    formula("attention weighted values:", "L * 2TTD", "all heads combined")
    formula("attention output proj:", "L * 2TDD")
    formula("SwiGLU W1/W3/W2:", "L * 3 * 2TDF")
    formula("LM head logits:", "2TDV")


def print_config(config: Config) -> None:
    param_parts = params(config)
    flop_parts = flops(config)
    total_params = sum(param_parts.values())
    total_flops = sum(flop_parts.values())

    print(f"\n{C.BOLD}{C.MAGENTA}{config.name}{C.RESET}")
    print(
        f"  layers={config.num_layers}, d_model={config.d_model}, heads={config.num_heads}, "
        f"d_head={config.d_head}, d_ff={config.d_ff}, ctx={config.context_length}"
    )
    print(
        f"  params={C.YELLOW}{human_count(total_params)}{C.RESET}, "
        f"fp32 memory={C.YELLOW}{human_bytes(total_params * BYTES_PER_FP32)}{C.RESET}, "
        f"forward matmul FLOPs={C.YELLOW}{human_count(total_flops)}{C.RESET}"
    )

    print_table(
        "Parameter breakdown",
        [
            (name, human_count(value), f"{value / total_params:6.2%}")
            for name, value in param_parts.items()
        ],
    )
    print_table(
        "Forward FLOPs breakdown",
        [
            (name, human_count(value), f"{value / total_flops:6.2%}")
            for name, value in flop_parts.items()
        ],
    )


def main() -> None:
    configs = [
        Config("GPT-2 small shaped", num_layers=12, d_model=768, num_heads=12),
        Config("GPT-2 medium shaped", num_layers=24, d_model=1024, num_heads=16),
        Config("GPT-2 large shaped", num_layers=36, d_model=1280, num_heads=20),
        Config("GPT-2 XL shaped", num_layers=48, d_model=1600, num_heads=25),
        Config(
            "GPT-2 XL shaped, long context",
            num_layers=48,
            d_model=1600,
            num_heads=25,
            context_length=16_384,
        ),
    ]

    print(f"{C.BOLD}Transformer LM resource accounting{C.RESET}")
    print(f"{C.DIM}Generated with Codex (GPT-5). Matrix multiply rule: 2*m*n*p FLOPs.{C.RESET}")
    print_formula_map()
    for config in configs:
        print_config(config)
    print(
        f"\n{C.BOLD}{C.YELLOW}Long-context note{C.RESET}\n"
        f"  Attention score/value matmuls scale as O(seq_len^2): each query attends over all keys.\n"
        f"  Increasing context from 1,024 to 16,384 is 16x longer, so these quadratic terms grow by 16^2 = 256x."
    )


if __name__ == "__main__":
    main()
