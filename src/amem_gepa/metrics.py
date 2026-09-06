"""Scoring for LoCoMo QA answers.

Not implemented yet -- milestone-2 work. Per docs/decisions/0004, this
should return per-category scores (not just one blended scalar) so GEPA's
Pareto search can be given multiple objectives instead of a single average.
"""

from __future__ import annotations


def score_answer(predicted: str, gold: str, category: str) -> float:
    """Single-instance score in [0, 1]. Adversarial-category instances score
    on correct refusal, not answer overlap -- don't reuse a plain string-match
    metric across all categories uninspected."""
    raise NotImplementedError


def bootstrap_ci(scores: list[float], n_resamples: int = 10_000) -> tuple[float, float]:
    """Bootstrap confidence interval for a headline aggregate metric
    (docs/decisions/0004) -- report alongside every test-set result, not a
    bare point estimate."""
    raise NotImplementedError
