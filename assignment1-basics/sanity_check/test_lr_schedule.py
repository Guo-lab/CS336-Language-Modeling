"""Manual sanity check for the cosine learning-rate schedule.

Run from the assignment root:
    python sanity_check/test_lr_schedule.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.optimizer import get_lr_cosine_schedule  # noqa: E402
from sanity_check.utils import title  # noqa: E402


def main() -> None:
    max_lr = 1.0
    min_lr = 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    print(title("Cosine LR schedule with warmup"))
    print(
        f"max_lr={max_lr}, min_lr={min_lr}, "
        f"warmup_iters={warmup_iters}, cosine_cycle_iters={cosine_cycle_iters}"
    )
    print("warmup: linearly rises; cosine: smoothly decays; post: stays at min_lr\n")

    for it in range(25):
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        if it < warmup_iters:
            phase = "warmup"
        elif it <= cosine_cycle_iters:
            phase = "cosine"
        else:
            phase = "post"
        bar = "#" * round(lr / max_lr * 36)
        print(f"{it:02d}  {phase:<6}  lr={lr:.6f}  {bar}")


if __name__ == "__main__":
    main()
