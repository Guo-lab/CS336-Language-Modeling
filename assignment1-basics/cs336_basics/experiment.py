from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .serialization import to_jsonable


class ExperimentLogger:
    """
    Lightweight local experiment logger.

    Directory layout:
        run_dir/
            config.json
            metrics.jsonl
            samples.jsonl
            checkpoints/
    """

    def __init__(
        self,
        run_name: str,
        out_dir: str | os.PathLike[str] = "artifacts/lm_experiments",
        config: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        self.run_name = _clean_run_name(run_name)
        self.root_dir = Path(out_dir)
        self.run_dir = self.root_dir / self.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.config_path = self.run_dir / "config.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.samples_path = self.run_dir / "samples.jsonl"
        self.start_time = time.time()

        if self.run_dir.exists() and not resume:
            raise FileExistsError(f"Experiment directory already exists: {self.run_dir}")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            self.save_config(config)

    def save_config(self, config: dict[str, Any]) -> None:
        payload = to_jsonable(config)

        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        record = {
            "step": step,
            "wall_clock_seconds": time.time() - self.start_time,
            **to_jsonable(metrics),
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def log_sample(self, step: int, prompt: str, completion: str) -> None:
        record = {
            "step": step,
            "prompt": prompt,
            "completion": completion,
            "wall_clock_seconds": time.time() - self.start_time,
        }
        with self.samples_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"step_{step:08d}.pt"


def _clean_run_name(run_name: str) -> str:
    cleaned = []
    for ch in run_name.strip().lower():
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    result = "".join(cleaned).strip("_")
    return result or "run"
