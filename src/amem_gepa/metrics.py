"""Scoring for LoCoMo QA answers.

Ported from the paper's own eval code (Maharana et al., ACL 2024) --
snap-research/locomo, `task_eval/evaluation.py` -- not a generic F1/ROUGE
implementation. Category 1 (multi-hop) uses a comma-split multi-answer F1;
categories 2/3/4 (temporal/open-domain/single-hop) use plain single-answer
F1; category 5 (adversarial) is scored on two literal refusal phrases, not
text overlap, since there's no gold answer to compare against -- its
`answer` field is null, the "trap" is in `adversarial_answer`
(docs/decisions/0004, 0010).

`docs/decisions/0009` and an earlier version of this file used ROUGE-L
because a secondary web summary (wrongly) described the paper's metric as
ROUGE-L -- corrected once the actual paper table was available
(docs/decisions/0010). `rouge_l_f1` is kept below since it's still
reasonable as a secondary signal, but `score_answer`/`score_batch` use the
paper-matching F1 by default.

Per docs/decisions/0004, report per-category, not just one blended number --
`score_batch` returns both.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from nltk.stem import PorterStemmer

from amem_gepa.datasets.locomo import CATEGORY_LABELS

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_ARTICLE_PATTERN = re.compile(r"\b(a|an|the|and)\b")
_PUNCTUATION = set(string.punctuation)
_stemmer = PorterStemmer()

# Category 5 (adversarial) scoring, verbatim from the paper's eval script:
# a literal substring check for one of these two refusal phrases, not text
# overlap with anything. Brittle by construction -- matches the paper's own
# known limitation, not something to "improve" on while the goal is fidelity.
_ADVERSARIAL_REFUSAL_PHRASES = ("no information available", "not mentioned")


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
    """Standard LCS-based ROUGE-L F1 (Lin, 2004). Not the paper's metric
    (see module docstring) -- kept as an available secondary signal, not
    used by score_answer/score_batch."""
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


def normalize_answer(s: str) -> str:
    """Paper's exact normalization (task_eval/evaluation.py:normalize_answer):
    strip commas, lowercase, strip punctuation, strip articles (a/an/the/and
    -- yes, "and" too, that's their regex not a typo here), collapse
    whitespace."""
    s = str(s).replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)
    s = _ARTICLE_PATTERN.sub(" ", s)
    return " ".join(s.split())


def _stemmed_tokens(text: str) -> list[str]:
    return [_stemmer.stem(w) for w in normalize_answer(text).split()]


def f1_score(prediction: str, ground_truth: str) -> float:
    """Single-answer F1 (task_eval/evaluation.py:f1_score): normalize +
    Porter-stem both sides, multiset (Counter) token overlap."""
    pred_tokens = _stemmed_tokens(prediction)
    gt_tokens = _stemmed_tokens(ground_truth)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_multi_answer(prediction: str, ground_truth: str) -> float:
    """Comma-split multi-answer F1 (task_eval/evaluation.py:f1), used only
    for category 1 (multi-hop) -- gold answers there are often several
    comma-separated facts (e.g. "pottery, camping, painting, swimming")
    drawn from different sessions. Splits both sides on commas, then for
    each gold sub-answer takes the best-matching predicted sub-phrase and
    averages across gold sub-answers."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    per_gt = [max(f1_score(p, gt) for p in predictions) for gt in ground_truths]
    return sum(per_gt) / len(per_gt)


def score_adversarial(prediction: str) -> float:
    """Paper's exact adversarial check (task_eval/evaluation.py:
    eval_question_answering, category 5 branch): 1.0 if the prediction
    contains one of two literal refusal phrases, 0.0 otherwise. Not text
    overlap with `adversarial_answer` -- the paper's own method doesn't
    look at the trap text at all, only whether the model said one of these
    specific things."""
    lowered = prediction.lower()
    return 1.0 if any(phrase in lowered for phrase in _ADVERSARIAL_REFUSAL_PHRASES) else 0.0


def score_answer(
    prediction: str,
    category: int,
    gold_answer: Optional[str] = None,
    adversarial_answer: Optional[str] = None,
) -> float:
    if category == 5:
        return score_adversarial(prediction)
    assert gold_answer is not None, f"category {category} requires gold_answer"
    gold_answer = str(gold_answer)
    if category == 3:
        # Paper quirk (task_eval/evaluation.py:eval_question_answering):
        # open-domain gold answers can carry semicolon-separated variants;
        # only the first is used as canonical.
        gold_answer = gold_answer.split(";")[0].strip()
    if category == 1:
        return f1_multi_answer(prediction, gold_answer)
    return f1_score(prediction, gold_answer)


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
