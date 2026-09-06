"""LoCoMo loading and the fixed train/val/test split (docs/decisions/0004).

Not implemented yet -- milestone-1 work: fetch the dataset, inspect its
actual conversation IDs/schema, then write `make_split()` and commit its
output as configs/locomo_split.json via `just split`
(scripts/make_locomo_split.py).
"""

from __future__ import annotations

from pathlib import Path

SPLIT_MANIFEST_PATH = Path("configs/locomo_split.json")


def load_split(split: str) -> list:
    """Return the LoCoMoInstance list (see gepa_adapter.py) for one of
    "train" / "val" / "test", using the conversation IDs pinned in
    configs/locomo_split.json."""
    raise NotImplementedError


def make_split(conversations: list, seed: int) -> dict[str, list[str]]:
    """Assign conversation IDs to train/val/test (5/2/3, docs/decisions/0004),
    balancing the 5 QA categories across splits rather than assigning purely
    at random. Returns {"train": [...], "val": [...], "test": [...]}."""
    raise NotImplementedError
