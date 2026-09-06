"""LoCoMo loading and the fixed train/val/test split (docs/decisions/0004).

Source: Maharana et al., "Evaluating Very Long-Term Conversational Memory of
LLM Agents" (ACL 2024) -- https://github.com/snap-research/locomo,
`data/locomo10.json`. Released for non-commercial research use; fetched at
runtime into a gitignored `data/` dir rather than committed, per its license
and to keep this repo from carrying someone else's dataset file. Only the
resulting conversation-ID split manifest (configs/locomo_split.json) is
committed.

Verified against the real file (see docs/01-related-work.md) rather than
trusting secondary summaries: 10 conversations, 1,986 QA pairs total,
5 categories via the integer `category` field:
    1 = single-hop, 2 = temporal, 3 = multi-hop, 4 = open-domain,
    5 = adversarial (answer is null; the "trap" answer is in
    `adversarial_answer`).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
RAW_PATH = Path("data/locomo10.json")
SPLIT_MANIFEST_PATH = Path("configs/locomo_split.json")

CATEGORY_LABELS = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}

TRAIN_SIZE = 5
VAL_SIZE = 2
TEST_SIZE = 3


@dataclass
class LoCoMoTurn:
    session: int
    dia_id: str
    speaker: str
    text: str
    date_time: str


@dataclass
class LoCoMoInstance:
    """One (conversation, question) pair, replayed against a fresh memory
    system built from `turns` before `question` is asked."""

    conversation_id: str
    turns: list[LoCoMoTurn]
    question: str
    gold_answer: Optional[str]
    category: int

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS[self.category]

    @property
    def is_adversarial(self) -> bool:
        return self.category == 5


def download_raw(dest: Path = RAW_PATH) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(LOCOMO_URL, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_raw_conversations(path: Path = RAW_PATH) -> list[dict]:
    download_raw(path)
    return json.loads(path.read_text())


def _turns_for_conversation(conv: dict) -> list[LoCoMoTurn]:
    turns = []
    # "session_<N>" holds the turn list; "session_<N>_date_time" and other
    # session_-prefixed keys are metadata -- match the plain two-part form
    # exactly rather than just checking the second segment is numeric (which
    # "session_1_date_time".split("_")[1] == "1" would also pass).
    session_keys = sorted(
        (k for k in conv if k.count("_") == 1 and k.split("_")[0] == "session" and k.split("_")[1].isdigit()),
        key=lambda k: int(k.split("_")[1]),
    )
    for key in session_keys:
        session_num = int(key.split("_")[1])
        date_time = conv.get(f"{key}_date_time", "")
        for turn in conv[key]:
            turns.append(
                LoCoMoTurn(
                    session=session_num,
                    dia_id=turn["dia_id"],
                    speaker=turn["speaker"],
                    text=turn["text"],
                    date_time=date_time,
                )
            )
    return turns


def category_counts_by_conversation(conversations: list[dict]) -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = {}
    for conv in conversations:
        per_cat: dict[int, int] = {}
        for qa in conv["qa"]:
            per_cat[qa["category"]] = per_cat.get(qa["category"], 0) + 1
        counts[conv["sample_id"]] = per_cat
    return counts


def _category_balance_score(
    assignment: dict[str, list[str]],
    counts: dict[str, dict[int, int]],
    overall_fraction: dict[int, float],
) -> float:
    score = 0.0
    for ids in assignment.values():
        split_counts: dict[int, int] = {}
        for conv_id in ids:
            for cat, n in counts[conv_id].items():
                split_counts[cat] = split_counts.get(cat, 0) + n
        split_total = sum(split_counts.values()) or 1
        for cat, target_frac in overall_fraction.items():
            split_frac = split_counts.get(cat, 0) / split_total
            score += (split_frac - target_frac) ** 2
    return score


def make_split(
    conversations: list[dict],
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
) -> dict[str, list[str]]:
    """Exhaustive search over conversation-level train/val/test assignments
    (docs/decisions/0004), minimizing per-category distribution deviation
    from the whole dataset. Deterministic -- no seed, no randomness: with
    only 10 conversations, all C(10,5)*C(5,2) = 2,520 assignments are cheap
    to score directly rather than approximated.
    """
    ids = [c["sample_id"] for c in conversations]
    if len(ids) != train_size + val_size + test_size:
        raise ValueError(
            f"Expected {train_size + val_size + test_size} conversations, got {len(ids)}"
        )

    counts = category_counts_by_conversation(conversations)
    overall_counts: dict[int, int] = {}
    for per_cat in counts.values():
        for cat, n in per_cat.items():
            overall_counts[cat] = overall_counts.get(cat, 0) + n
    overall_total = sum(overall_counts.values())
    overall_fraction = {cat: n / overall_total for cat, n in overall_counts.items()}

    best_assignment: Optional[dict[str, list[str]]] = None
    best_score = float("inf")
    for train_ids in itertools.combinations(ids, train_size):
        remaining = [i for i in ids if i not in train_ids]
        for val_ids in itertools.combinations(remaining, val_size):
            test_ids = [i for i in remaining if i not in val_ids]
            assignment = {
                "train": list(train_ids),
                "val": list(val_ids),
                "test": test_ids,
            }
            score = _category_balance_score(assignment, counts, overall_fraction)
            if score < best_score:
                best_score = score
                best_assignment = assignment

    assert best_assignment is not None
    return best_assignment


def load_split_manifest(path: Path = SPLIT_MANIFEST_PATH) -> dict[str, list[str]]:
    return json.loads(path.read_text())


def load_split(
    split: str,
    manifest_path: Path = SPLIT_MANIFEST_PATH,
    raw_path: Path = RAW_PATH,
) -> list[LoCoMoInstance]:
    """Return every (conversation, question) pair for one of
    "train" / "val" / "test", per the committed split manifest."""
    manifest = load_split_manifest(manifest_path)
    if split not in manifest:
        raise ValueError(f"Unknown split {split!r}, expected one of {list(manifest)}")
    wanted_ids = set(manifest[split])

    instances = []
    for conv in load_raw_conversations(raw_path):
        if conv["sample_id"] not in wanted_ids:
            continue
        turns = _turns_for_conversation(conv["conversation"])
        for qa in conv["qa"]:
            instances.append(
                LoCoMoInstance(
                    conversation_id=conv["sample_id"],
                    turns=turns,
                    question=qa["question"],
                    gold_answer=qa.get("answer"),
                    category=qa["category"],
                )
            )
    return instances
