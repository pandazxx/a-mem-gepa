"""Scoring for LoCoMo QA answers.

The paper reports ROUGE-L on LoCoMo (docs/01-related-work.md), so categories
1-4 (single-hop, temporal, multi-hop, open-domain) are scored with ROUGE-L
F1 against the gold answer. Category 5 (adversarial) has no gold answer to
compare against -- its `answer` field is null and the "trap" is in
`adversarial_answer` (docs/decisions/0004) -- so it's scored on whether the
system avoided confidently reproducing that trap answer, not on text overlap
with a reference.

Per docs/decisions/0004, report per-category, not just one blended number --
`score_batch` returns both.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from amem_gepa.datasets.locomo import CATEGORY_LABELS

ADVERSARIAL_OVERLAP_THRESHOLD = 0.5

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if token_a == token_b else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(str(text).lower())


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Standard LCS-based ROUGE-L F1 (Lin, 2004), lowercased and split on
    alphanumeric runs -- NOT plain `.split()` (see git history: that version
    tokenized on whitespace only, so a real "...is transgender." never
    matched gold "Transgender woman" because the trailing period made
    "transgender." a different token from "transgender"; a full run against
    real Llama 3.2:1b baseline output showed this roughly doubled every
    category's score once fixed, since chatty full-sentence answers very
    often have the answer word followed by punctuation). Not necessarily
    bit-identical to whatever toolkit the paper used -- fine for a sanity
    check, not for claiming an exact reproduction (docs/decisions/0006)."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_adversarial(prediction: str, adversarial_answer: str) -> float:
    """1.0 if the system avoided confidently asserting the trap answer,
    0.0 if it reproduced it. Heuristic (docs/decisions/0006): treat high
    text overlap with `adversarial_answer` as having fallen for the trap,
    regardless of phrasing. Revisit this threshold once real outputs from
    milestone-2's reproduction run are available -- it hasn't been tuned
    against real model outputs yet."""
    overlap = rouge_l_f1(prediction, adversarial_answer)
    return 0.0 if overlap >= ADVERSARIAL_OVERLAP_THRESHOLD else 1.0


def score_answer(
    prediction: str,
    category: int,
    gold_answer: Optional[str] = None,
    adversarial_answer: Optional[str] = None,
) -> float:
    if category == 5:
        assert adversarial_answer is not None, "category 5 requires adversarial_answer"
        return score_adversarial(prediction, adversarial_answer)
    assert gold_answer is not None, f"category {category} requires gold_answer"
    return rouge_l_f1(prediction, str(gold_answer))


@dataclass
class ScoreSummary:
    mean: float
    n: int
    ci_low: float
    ci_high: float


def bootstrap_ci(scores: list[float], n_resamples: int = 10_000, ci: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `scores` (docs/decisions/0004
    -- always report this alongside a headline number, never a bare point
    estimate)."""
    if not scores:
        return (0.0, 0.0)
    n = len(scores)
    if n == 1:
        return (scores[0], scores[0])
    means = []
    for _ in range(n_resamples):
        resample_sum = sum(scores[random.randrange(n)] for _ in range(n))
        means.append(resample_sum / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = min(int((1 + ci) / 2 * n_resamples), n_resamples - 1)
    return means[lo_idx], means[hi_idx]


def summarize(scores: list[float], n_resamples: int = 10_000) -> ScoreSummary:
    mean = sum(scores) / len(scores) if scores else 0.0
    ci_low, ci_high = bootstrap_ci(scores, n_resamples=n_resamples)
    return ScoreSummary(mean=mean, n=len(scores), ci_low=ci_low, ci_high=ci_high)


def score_batch(
    predictions: list[str],
    categories: list[int],
    gold_answers: list[Optional[str]],
    adversarial_answers: list[Optional[str]],
    n_resamples: int = 10_000,
) -> dict[str, ScoreSummary]:
    """Per-category + aggregate summaries for one batch of predictions."""
    by_category: dict[int, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for pred, cat, gold, adv in zip(predictions, categories, gold_answers, adversarial_answers):
        score = score_answer(pred, cat, gold_answer=gold, adversarial_answer=adv)
        by_category[cat].append(score)
        all_scores.append(score)

    result = {"aggregate": summarize(all_scores, n_resamples=n_resamples)}
    for cat, label in CATEGORY_LABELS.items():
        if cat in by_category:
            result[label] = summarize(by_category[cat], n_resamples=n_resamples)
    return result
