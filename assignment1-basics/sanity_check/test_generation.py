"""Manual sanity check for decoding helpers.

Run from the assignment root:
    python sanity_check/test_generation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.generation import decode, sample_next_token  # noqa: E402
from sanity_check.utils import pass_text, title, warn_text  # noqa: E402


class TinyStepModel(torch.nn.Module):
    def __init__(self, vocab_size: int, scripted_next_tokens: list[int]) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.scripted_next_tokens = scripted_next_tokens
        self.calls = 0
        self.input_lengths: list[int] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.input_lengths.append(token_ids.shape[-1])
        next_token = self.scripted_next_tokens[
            min(self.calls, len(self.scripted_next_tokens) - 1)
        ]
        self.calls += 1

        logits = torch.full(
            (*token_ids.shape, self.vocab_size), -20.0, device=token_ids.device
        )
        logits[..., -1, next_token] = 20.0
        return logits


def main() -> None:
    print(title("Generation sanity check"))
    print("TinyStepModel returns very high logits for scripted next-token IDs.")
    print("This checks decoding mechanics, not text quality.\n")
    print(warn_text("Note: CHECK_LOG=1 disables ANSI colors in saved logs.\n"))

    torch.manual_seed(0)
    logits = torch.tensor([[0.1, 8.0, 0.2, 2.0]])
    expected_shape = (1, 1)
    expected_token = [[1]]
    next_token = sample_next_token(logits, temperature=1.0, top_p=0.01)
    assert next_token.shape == expected_shape
    assert next_token.tolist() == expected_token
    print(f"Case 1: {pass_text()} top-p keeps at least the highest-probability token")
    print(f"  logits:         {logits.tolist()}")
    print("  top_p:          0.01")
    print(f"  expected shape: {expected_shape}")
    print(f"  actual shape:   {tuple(next_token.shape)}")
    print(f"  expected token: {expected_token}")
    print(f"  actual token:   {next_token.tolist()}\n")

    torch.manual_seed(1)
    batched_logits = torch.tensor([[0.1, 3.0, 0.2, 2.0], [4.0, 0.1, 0.2, 0.3]])
    sampled = sample_next_token(batched_logits, temperature=1.0, top_p=None)
    assert sampled.shape == (2, 1)
    assert torch.all((0 <= sampled) & (sampled < batched_logits.shape[-1]))
    print(f"Case 2: {pass_text()} plain multinomial sampling supports batched logits")
    print(f"  logits:         {batched_logits.tolist()}")
    print(f"  logits shape:   {tuple(batched_logits.shape)}")
    print("  expected shape: (2, 1)")
    print(f"  actual shape:   {tuple(sampled.shape)}")
    print(f"  sampled tokens: {sampled.tolist()}\n")

    eos_token_id = 2
    model = TinyStepModel(vocab_size=5, scripted_next_tokens=[1, eos_token_id])
    prompt = torch.tensor([[0, 4, 4]])
    generated = decode(
        model=model,
        prompt_token_ids=prompt,
        max_new_tokens=5,
        eos_token_id=eos_token_id,
        temperature=1.0,
        top_p=0.9,
    )
    expected_generated = [[0, 4, 4, 1, eos_token_id]]
    assert generated.tolist() == expected_generated
    print(f"Case 3: {pass_text()} decode appends tokens until eos")
    print(f"  prompt:         {prompt.tolist()}")
    print("  scripted next:  [1, 2]")
    print("  fake logits:    selected token gets 20.0, all others get -20.0")
    print(f"  expected:       {expected_generated}")
    print(f"  actual:         {generated.tolist()}\n")

    model = TinyStepModel(vocab_size=5, scripted_next_tokens=[1, 3, 4])
    prompt = torch.tensor([[0, 4]])
    generated = decode(model=model, prompt_token_ids=prompt, max_new_tokens=3)
    expected_generated = [[0, 4, 1, 3, 4]]
    assert generated.tolist() == expected_generated
    print(f"Case 4: {pass_text()} decode respects max_new_tokens when eos is absent")
    print(f"  prompt:         {prompt.tolist()}")
    print("  scripted next:  [1, 3, 4]")
    print("  fake logits:    selected token gets 20.0, all others get -20.0")
    print(f"  expected:       {expected_generated}")
    print(f"  actual:         {generated.tolist()}\n")

    model = TinyStepModel(vocab_size=5, scripted_next_tokens=[1])
    prompt = torch.tensor([[0, 4]])
    generated = decode(model=model, prompt_token_ids=prompt, max_new_tokens=0)
    assert generated.tolist() == prompt.tolist()
    assert model.calls == 0
    print(f"Case 5: {pass_text()} max_new_tokens=0 returns the prompt without model calls")
    print(f"  prompt:         {prompt.tolist()}")
    print("  fake logits:    not produced because model should not be called")
    print(f"  expected:       {prompt.tolist()}")
    print(f"  actual:         {generated.tolist()}")
    print(f"  model calls:    {model.calls}\n")

    model = TinyStepModel(vocab_size=5, scripted_next_tokens=[1, 1])
    long_prompt = torch.tensor([[0, 1, 2, 3, 4]])
    context_length = 3
    _ = decode(
        model=model,
        prompt_token_ids=long_prompt,
        max_new_tokens=2,
        context_length=context_length,
    )
    assert max(model.input_lengths) <= context_length
    print(f"Case 6: {pass_text()} context_length crops model input while keeping full output history")
    print("  fake logits:            selected token gets 20.0, all others get -20.0")
    print(f"  prompt length:          {long_prompt.shape[-1]}")
    print(f"  context_length:         {context_length}")
    print(f"  expected max input len: {context_length}")
    print(f"  actual input lengths:   {model.input_lengths}")


if __name__ == "__main__":
    main()
