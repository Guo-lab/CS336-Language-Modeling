"""Small opt-in debugging helpers."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}


class NoOpDebugger:
    """Callable placeholder used when debugging is disabled."""

    def __call__(self, message: str, *parts: Any, **fields: Any) -> None:
        return None


class Debugger:
    """Write human-readable debug records for one source file."""

    def __init__(self, file_path: str | os.PathLike[str]) -> None:
        self.file_path = Path(file_path)
        self.file_name = self.file_path.name
        self.run_id = os.environ.get("CS336_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(os.environ.get("CS336_LOG_DIR", "logs"))
        self.log_path = self.log_dir / f"{self.run_id}_debug.log"

    def __call__(self, message: str, *parts: Any, **fields: Any) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        details = _format_details(parts, fields)
        line = f"[{timestamp}] {self.file_name} | {message}"
        if details:
            line = f"{line} | {details}"

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            return None


def get_debugger(file_path: str | os.PathLike[str]) -> Debugger | NoOpDebugger:
    """Return a file-scoped debugger, or a no-op callable when disabled."""

    if os.environ.get("CS336_DEBUG", "").lower() not in _TRUTHY:
        return NoOpDebugger()

    file_name = Path(file_path).name
    enabled_files = os.environ.get("CS336_DEBUG_FILES", "").strip()
    if enabled_files and enabled_files != "*":
        allowed_files = {item.strip() for item in enabled_files.split(",") if item.strip()}
        if file_name not in allowed_files:
            return NoOpDebugger()

    return Debugger(file_path)


def _format_details(parts: tuple[Any, ...], fields: dict[str, Any]) -> str:
    rendered_parts = [str(part) for part in parts]
    rendered_fields = [f"{key}={value!r}" for key, value in fields.items()]
    return " | ".join(rendered_parts + rendered_fields)
