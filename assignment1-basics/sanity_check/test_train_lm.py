"""Manual sanity check for the LM training script.

Run from the assignment root:
    python sanity_check/test_train_lm.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
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


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    print(title("Train LM sanity check"))
    print("Creates patterned token .npy files, runs a tiny train loop, and checks outputs.")
    print("The pattern is 0, 1, ..., 15 repeated, so loss should decrease.\n")

    run_root = ROOT / "artifacts" / "lm_experiments"
    run_name = f"sanity_check_{int(time.time())}"
    run_dir = run_root / run_name
    tmp_dir = tempfile.TemporaryDirectory(prefix="train_lm_sanity_")
    train_path, valid_path = write_tiny_data(Path(tmp_dir.name))

    cmd = [
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
    ]

    print("Command:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)

    print("\nScript stdout:")
    print(result.stdout.rstrip())
    if result.stderr:
        print("\nScript stderr:")
        print(result.stderr.rstrip())

    if result.returncode != 0:
        print(fail_text(f"\nFAIL train_lm.py exited with {result.returncode}"))
        return result.returncode

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_path = run_dir / "checkpoints" / "step_00000050.pt"

    expected_paths = [config_path, metrics_path, checkpoint_path]
    for path in expected_paths:
        if not path.exists():
            print(fail_text(f"\nFAIL missing expected output: {path.relative_to(ROOT)}"))
            return 1

    metrics = read_jsonl(metrics_path)
    has_train_loss = any("train_loss" in record for record in metrics)
    has_valid_loss = any("valid_loss" in record for record in metrics)
    has_lr = any("lr" in record for record in metrics)
    if not (has_train_loss and has_valid_loss and has_lr):
        print(fail_text("\nFAIL metrics.jsonl is missing train_loss, valid_loss, or lr"))
        print(metrics)
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
    print(f"Checkpoint path: {checkpoint_path.relative_to(ROOT)}")
    print(f"Metric records:  {len(metrics)}")
    print(f"First metric:    {metrics[0]}")
    print(f"Train loss:      {first_train_loss:.4f} -> {last_train_loss:.4f}")
    print(pass_text("\nPASS train_lm.py runs, learns a tiny pattern, logs metrics, and saves a checkpoint"))
    tmp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
