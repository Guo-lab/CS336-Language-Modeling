from __future__ import annotations

import os
from typing import Any


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def _style(text: str, *codes: str) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    return "".join(codes) + text + _Ansi.RESET


def header(text: str) -> None:
    print(_style(text, _Ansi.BOLD, _Ansi.CYAN))


def info(label: str, value: Any) -> None:
    print(f"{_style(label + ':', _Ansi.GREEN)} {value}")


def warn(label: str, value: Any) -> None:
    print(f"{_style(label + ':', _Ansi.YELLOW)} {value}")


def error(label: str, value: Any) -> None:
    print(f"{_style(label + ':', _Ansi.RED)} {value}")


def dim(text: str) -> str:
    return _style(text, _Ansi.DIM)
