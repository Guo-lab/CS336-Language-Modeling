"""Manual sanity check for the LM training script.

Run from the assignment root:
    python sanity_check/test_train_lm.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sanity_check.utils import fail_text, pass_text, title  # noqa: E402


def write_tiny_data(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = np.arange(16, dtype=np.uint16)

    train_path = out_dir / "tiny_train.npy"
    valid_path = out_dir / "tiny_valid.npy"
    np.save(train_path, np.tile(pattern, 512))
    np.save(valid_path, np.tile(pattern, 128))
    return train_path, valid_path


def write_tiny_tokenizer(out_dir: Path) -> Path:
    tokenizer_dir = out_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    vocab = {
        str(token_id): {"hex": bytes([ord("a") + token_id]).hex()}
        for token_id in range(16)
    }
    with (tokenizer_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f)
    with (tokenizer_dir / "merges.json").open("w", encoding="utf-8") as f:
        json.dump([], f)
    return tokenizer_dir


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_command(cmd: list[str], label: str) -> subprocess.CompletedProcess[str]:
    print(f"{label} command:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)

    print(f"\n{label} stdout:")
    print(result.stdout.rstrip())
    if result.stderr:
        print(f"\n{label} stderr:")
        print(result.stderr.rstrip())
    return result


def main() -> int:
    print(title("Train LM sanity check"))
    print("Creates patterned token .npy files, runs a tiny train loop, and checks outputs.")
    print("The pattern is 0, 1, ..., 15 repeated, so loss should decrease.\n")

    run_root = ROOT / "artifacts" / "lm_experiments"
    run_name = "sanity_check"
    run_dir = run_root / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    tmp_dir = tempfile.TemporaryDirectory(prefix="train_lm_sanity_")
    tmp_path = Path(tmp_dir.name)
    train_path, valid_path = write_tiny_data(tmp_path)
    tokenizer_dir = write_tiny_tokenizer(tmp_path)

    train_cmd = [
        sys.executable,
        "scripts/train_lm.py",
        "--train-data",
        str(train_path),
        "--valid-data",
        str(valid_path),
        "--vocab-size",
        "16",
        "--run-name",
        run_name,
        "--out-dir",
        str(run_root.relative_to(ROOT)),
        "--context-length",
        "8",
        "--batch-size",
        "2",
        "--num-layers",
        "1",
        "--d-model",
        "16",
        "--num-heads",
        "4",
        "--d-ff",
        "64",
        "--max-iters",
        "50",
        "--warmup-iters",
        "5",
        "--cosine-cycle-iters",
        "50",
        "--log-every",
        "5",
        "--eval-every",
        "25",
        "--eval-iters",
        "1",
        "--save-every",
        "50",
        "--max-lr",
        "0.01",
        "--min-lr",
        "0.001",
        "--tokenizer",
        str(tokenizer_dir),
        "--sample-prompt",
        "abc",
        "--sample-every",
        "25",
        "--sample-max-new-tokens",
        "8",
        "--sample-temperature",
        "1.0",
        "--sample-top-p",
        "0.9",
        "--wandb-project",
        "cs336-sanity",
        "--wandb-mode",
        "disabled",
    ]

    result = run_command(train_cmd, "Initial train")

    if result.returncode != 0:
        print(fail_text(f"\nFAIL train_lm.py exited with {result.returncode}"))
        return result.returncode

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    samples_path = run_dir / "samples.jsonl"
    checkpoint_path = run_dir / "checkpoints" / "step_00000050.pt"

    expected_paths = [config_path, metrics_path, samples_path, checkpoint_path]
    for path in expected_paths:
        if not path.exists():
            print(fail_text(f"\nFAIL missing expected output: {path.relative_to(ROOT)}"))
            return 1

    metrics = read_jsonl(metrics_path)
    samples = read_jsonl(samples_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["logging"]["wandb_project"] != "cs336-sanity":
        print(fail_text("\nFAIL config.json did not record wandb_project"))
        return 1
    if config["logging"]["wandb_mode"] != "disabled":
        print(fail_text("\nFAIL config.json did not record wandb_mode"))
        return 1
    has_train_loss = any("train_loss" in record for record in metrics)
    has_valid_loss = any("valid_loss" in record for record in metrics)
    has_lr = any("lr" in record for record in metrics)
    if not (has_train_loss and has_valid_loss and has_lr):
        print(fail_text("\nFAIL metrics.jsonl is missing train_loss, valid_loss, or lr"))
        print(metrics)
        return 1
    if not samples or not all("prompt" in record and "completion" in record for record in samples):
        print(fail_text("\nFAIL samples.jsonl is missing prompt/completion records"))
        print(samples)
        return 1

    train_losses = [record["train_loss"] for record in metrics if "train_loss" in record]
    first_train_loss = train_losses[0]
    last_train_loss = train_losses[-1]
    if last_train_loss >= first_train_loss * 0.8:
        print(fail_text("\nFAIL train loss did not decrease enough on the patterned data"))
        print(f"first train loss: {first_train_loss:.4f}")
        print(f"last train loss:  {last_train_loss:.4f}")
        return 1

    print(f"\nRun directory:   {run_dir.relative_to(ROOT)}")
    print(f"Config path:     {config_path.relative_to(ROOT)}")
    print(f"Metrics path:    {metrics_path.relative_to(ROOT)}")
    print(f"Samples path:    {samples_path.relative_to(ROOT)}")
    print(f"Checkpoint path: {checkpoint_path.relative_to(ROOT)}")
    print(f"Metric records:  {len(metrics)}")
    print(f"Sample records:  {len(samples)}")
    print(f"First metric:    {metrics[0]}")
    print(f"First sample:    {samples[0]}")
    print(f"Train loss:      {first_train_loss:.4f} -> {last_train_loss:.4f}")

    resume_cmd = train_cmd.copy()
    resume_cmd[resume_cmd.index("--max-iters") + 1] = "105"
    resume_cmd.extend(["--resume-from", str(checkpoint_path)])

    print()
    resume_result = run_command(resume_cmd, "Resume train")
    if resume_result.returncode != 0:
        print(fail_text(f"\nFAIL train_lm.py resume exited with {resume_result.returncode}"))
        return resume_result.returncode

    resumed_checkpoint_path = run_dir / "checkpoints" / "step_00000105.pt"
    if not resumed_checkpoint_path.exists():
        print(
            fail_text(
                f"\nFAIL missing resumed checkpoint: {resumed_checkpoint_path.relative_to(ROOT)}"
            )
        )
        return 1

    resumed_metrics = read_jsonl(metrics_path)
    resumed_samples = read_jsonl(samples_path)
    if not any(record.get("step") == 105 for record in resumed_metrics):
        print(fail_text("\nFAIL metrics.jsonl does not include resumed step 105"))
        return 1
    step50_train_loss = next(
        record["train_loss"]
        for record in resumed_metrics
        if record.get("step") == 50 and "train_loss" in record
    )
    step105_train_loss = next(
        record["train_loss"]
        for record in resumed_metrics
        if record.get("step") == 105 and "train_loss" in record
    )
    if step105_train_loss >= step50_train_loss:
        print(fail_text("\nFAIL train loss did not keep decreasing after resume"))
        print(f"step 50 train loss: {step50_train_loss:.4f}")
        print(f"step 105 train loss: {step105_train_loss:.4f}")
        return 1
    if not any(record.get("step") == 105 for record in resumed_samples):
        print(fail_text("\nFAIL samples.jsonl does not include resumed step 105"))
        return 1

    print(f"Resumed checkpoint path: {resumed_checkpoint_path.relative_to(ROOT)}")
    print(f"Metric records after resume: {len(resumed_metrics)}")
    print(f"Sample records after resume: {len(resumed_samples)}")
    print(f"Resume train loss: {step50_train_loss:.4f} -> {step105_train_loss:.4f}")
    print(
        pass_text(
            "\nPASS train_lm.py trains, learns a tiny pattern, logs metrics, saves, and resumes"
        )
    )
    tmp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
