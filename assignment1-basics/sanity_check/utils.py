"""Shared helpers for manual sanity checks."""

from __future__ import annotations

import os

BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def color(text: str, code: str) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    return f"{code}{text}{RESET}"


def title(text: str) -> str:
    return color(text, BLUE + BOLD)


def pass_text(text: str = "PASS") -> str:
    return color(text, GREEN + BOLD)


def fail_text(text: str = "FAIL") -> str:
    return color(text, RED + BOLD)


def warn_text(text: str) -> str:
    return color(text, YELLOW + BOLD)
