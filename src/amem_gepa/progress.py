"""Progress reporting for long-running eval loops. A single LoCoMo baseline
run against a slow local model can take many hours (observed: 12+ hours,
still incomplete, on Llama 3.2:1b/M3 Pro) -- print one line per step so
there's a visible rate/ETA instead of a silent terminal.
"""

from __future__ import annotations

import sys
import time


def format_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "?"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressReporter:
    def __init__(self, total: int, label: str, done: int = 0, out=sys.stderr):
        self.total = total
        self.label = label
        self.done = done
        self.out = out
        self.start = time.monotonic()

    def tick(self, note: str = "") -> None:
        self.done += 1
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining_items = max(self.total - self.done, 0)
        eta = remaining_items / rate if rate > 0 else float("inf")
        suffix = f" {note}" if note else ""
        print(
            f"[{self.label}] {self.done}/{self.total} "
            f"({rate:.3f}/s, elapsed={format_duration(elapsed)}, "
            f"eta={format_duration(eta)}){suffix}",
            file=self.out,
        )
