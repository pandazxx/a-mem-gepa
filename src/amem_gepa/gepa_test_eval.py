"""Held-out test comparison: the paper's baseline prompts vs. a GEPA candidate,
both run through the paper-reproduction pipeline via the same adapter that
trained the candidate (docs/decisions/0018).

Why a separate tool: `just eval` drives the legacy `agiresearch/A-mem`
library pipeline (2 prompts, JSON evolution, our own QA prompt, metrics.py)
and cannot run a 4-prompt repro-pipeline candidate -- see
docs/02-training-explained.md section 8. `just reproduce` runs the paper's
prompts on the full dataset with no candidate injection and no split.

What this does, per arm ("baseline" = prompts/baseline_repro, "candidate" =
the directory passed in): for every test conversation, build memory with the
arm's prompts (AMemGEPAAdapter.build_for_conversation: injection, build
cache, mid-build checkpoints, think=False, progress lines), then answer EVERY
question of that conversation -- not the stratified training subset -- and
score it with the vendored `calculate_metrics` suite (the paper's own
metrics, docs/decisions/0013). Category 5 is reported both ways
(docs/decisions/0015): the paper's reading (metrics against the trap text)
and the refusal reading (picked "Not mentioned in the conversation").

Resumable per question (`qa_progress_<arm>.jsonl`), same mechanism and
reasoning as paper_repro's QA progress log.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from amem_gepa.datasets.locomo import CATEGORY_LABELS, LoCoMoInstance
from amem_gepa.gepa_adapter import REFUSAL_OPTION, AMemGEPAAdapter, reference_answer
from amem_gepa.paper_repro import _ProgressTracker, ensure_repro_repo_importable

logger = logging.getLogger("amem_gepa.gepa_test_eval")

BASELINE_ARM = "baseline"
CANDIDATE_ARM = "candidate"

# Same zero-row paper_repro.run_full_reproduction uses when there is no
# reference to score against, so aggregates stay comparable.
EMPTY_METRICS = {
    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0,
    "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "bert_f1": 0.0,
    "meteor": 0.0, "sbert_similarity": 0.0,
}  # fmt: skip


@dataclass
class QuestionResult:
    arm: str
    conversation_id: str
    q_idx: int
    category: int
    question: str
    prediction: str
    reference: str
    metrics: dict
    # Category 5 only: 1.0 if the model picked the refusal option. None otherwise.
    refusal: float | None = None
    error: str | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.conversation_id, self.q_idx)


# ---------------------------------------------------------------------------
# per-question progress log (resume)
# ---------------------------------------------------------------------------


def progress_path(results_dir: Path, arm: str) -> Path:
    return results_dir / f"qa_progress_{arm}.jsonl"


def load_progress(path: Path) -> dict[tuple[str, int], QuestionResult]:
    done: dict[tuple[str, int], QuestionResult] = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        result = QuestionResult(**json.loads(line))
        done[result.key] = result  # last line wins
    return done


def append_progress(path: Path, result: QuestionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------


def _by_conversation(instances: list[LoCoMoInstance]) -> dict[str, list[LoCoMoInstance]]:
    out: dict[str, list[LoCoMoInstance]] = {}
    for inst in instances:
        out.setdefault(inst.conversation_id, []).append(inst)
    return out


def evaluate_arm(
    adapter: AMemGEPAAdapter,
    arm: str,
    candidate: dict[str, str],
    instances: list[LoCoMoInstance],
    results_dir: Path,
    log_every: int = 10,
) -> list[QuestionResult]:
    """Build each conversation once with `candidate`, answer every question,
    score with the paper's suite. Resumes from results_dir/qa_progress_<arm>.jsonl."""
    ensure_repro_repo_importable()
    from llm_text_parsers import parse_plain_text_answer
    from utils import calculate_metrics

    path = progress_path(results_dir, arm)
    done = load_progress(path)
    if done:
        logger.info("[%s] resuming: %d questions already answered", arm, len(done))
    tracker = _ProgressTracker(logger, f"[{arm}] QA", len(instances))
    results: list[QuestionResult] = []

    for conv_id, conv_instances in _by_conversation(instances).items():
        pending = []
        for q_idx, q in enumerate(conv_instances):
            cached = done.get((conv_id, q_idx))
            if cached is not None:
                results.append(cached)
                tracker.record(0.0, cached=True)
            else:
                pending.append((q_idx, q))
        if not pending:
            continue

        agent, _trace, cache_hit, build_error, renoted = adapter.agent_for_questions(
            candidate, conv_id, conv_instances[0].turns, [q for _, q in pending]
        )
        logger.info(
            "[%s] %s: memory %s, %d questions to answer",
            arm,
            conv_id,
            f"PROXY -- seed build with {renoted} notes re-constructed (docs/decisions/0020)"
            if renoted
            else ("loaded from cache" if cache_hit else ("build FAILED" if build_error else "built")),
            len(pending),
        )

        for q_idx, q in pending:
            started = time.monotonic()
            reference = reference_answer(q)
            if build_error is not None:
                result = QuestionResult(
                    arm, conv_id, q_idx, q.category, q.question, "", reference, dict(EMPTY_METRICS),
                    refusal=0.0 if q.is_adversarial else None,
                    error=f"memory build failed: {build_error}",
                )  # fmt: skip
            else:
                try:
                    raw, _prompt, _context = agent.answer_question(q.question, q.category, reference)
                    prediction = parse_plain_text_answer(raw)
                    error = None
                except Exception as e:  # noqa: BLE001 -- one bad question must not kill the arm
                    prediction, error = "", str(e)
                metrics = (
                    dict(calculate_metrics(prediction, reference))
                    if reference and not error
                    else dict(EMPTY_METRICS)
                )
                refusal = None
                if q.is_adversarial:
                    refusal = 1.0 if REFUSAL_OPTION in prediction.lower() else 0.0
                result = QuestionResult(
                    arm, conv_id, q_idx, q.category, q.question, prediction, reference, metrics,
                    refusal=refusal, error=error,
                )  # fmt: skip
            append_progress(path, result)
            results.append(result)
            tracker.record(time.monotonic() - started, cached=False)
            if tracker.completed_units % log_every == 0 or tracker.completed_units == len(instances):
                logger.info(tracker.progress_line(extra=f"conv={conv_id} cat={q.category}"))
    return results


# ---------------------------------------------------------------------------
# aggregation + paired bootstrap
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_arm(results: list[QuestionResult]) -> dict:
    """Mean of every metric overall and per category, plus the refusal rate
    for category 5 and the error count. Same means `utils.aggregate_metrics`
    reports as 'mean'; computed here so the tool has no dependency on the
    vendored aggregation beyond the metrics themselves."""
    metric_names = sorted({name for r in results for name in r.metrics})
    buckets: dict[str, list[QuestionResult]] = {"overall": list(results)}
    for r in results:
        buckets.setdefault(CATEGORY_LABELS.get(r.category, f"category_{r.category}"), []).append(r)
    summary: dict = {}
    for label, rs in buckets.items():
        entry = {"n": len(rs), "errors": sum(1 for r in rs if r.error)}
        for name in metric_names:
            entry[name] = _mean([float(r.metrics.get(name, 0.0)) for r in rs])
        refusals = [r.refusal for r in rs if r.refusal is not None]
        if refusals:
            entry["refusal_rate"] = _mean(refusals)
        summary[label] = entry
    return summary


def paired_bootstrap(
    baseline: list[QuestionResult],
    candidate: list[QuestionResult],
    metric: str = "f1",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired (same question in both arms) bootstrap of the candidate-minus-
    baseline mean difference, overall and per category. Paired because both
    arms answer the identical question set; resampling questions with
    replacement and recomputing the mean difference gives a 95% percentile
    interval that already accounts for question-level variance (docs/
    decisions/0004 asks for a CI on the headline number, not a bare point)."""
    import numpy as np

    base_by_key = {r.key: r for r in baseline}
    rng = np.random.default_rng(seed)
    groups: dict[str, list[tuple[float, float]]] = {"overall": []}
    for c in candidate:
        b = base_by_key.get(c.key)
        if b is None:
            continue
        if metric == "refusal_rate":
            if c.refusal is None or b.refusal is None:
                continue
            pair = (float(b.refusal), float(c.refusal))
        else:
            pair = (float(b.metrics.get(metric, 0.0)), float(c.metrics.get(metric, 0.0)))
        groups["overall"].append(pair)
        groups.setdefault(CATEGORY_LABELS.get(c.category, f"category_{c.category}"), []).append(pair)

    out: dict = {"metric": metric, "resamples": n_resamples}
    for label, pairs in groups.items():
        if not pairs:
            continue
        b = np.array([p[0] for p in pairs])
        c = np.array([p[1] for p in pairs])
        d = c - b
        idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
        means = d[idx].mean(axis=1)
        out[label] = {
            "n": len(d),
            "baseline_mean": float(b.mean()),
            "candidate_mean": float(c.mean()),
            "diff_mean": float(d.mean()),
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
        }
    return out


def format_comparison(summary: dict[str, dict], bootstrap: dict, refusal_bootstrap: dict | None) -> str:
    """Markdown table in the paper's category order, ready to paste into a
    docs/experiments entry."""
    order = [CATEGORY_LABELS[c] for c in sorted(CATEGORY_LABELS)] + ["overall"]
    base, cand = summary.get(BASELINE_ARM, {}), summary.get(CANDIDATE_ARM)

    def fmt(x: float | None) -> str:
        return "--" if x is None else f"{x:.4f}"

    lines = []
    if cand is None:
        lines.append("| category | n | F1 | BLEU-1 |")
        lines.append("|---|---|---|---|")
        for label in order:
            e = base.get(label)
            if e:
                lines.append(f"| {label} | {e['n']} | {fmt(e.get('f1'))} | {fmt(e.get('bleu1'))} |")
        return "\n".join(lines)

    lines.append("| category | n | baseline F1 | candidate F1 | delta F1 [95% CI] | baseline BLEU-1 | candidate BLEU-1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in order:
        b, c, bs = base.get(label), cand.get(label), bootstrap.get(label)
        if not (b and c):
            continue
        delta = f"{bs['diff_mean']:+.4f} [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]" if bs else "--"
        lines.append(
            f"| {label} | {c['n']} | {fmt(b.get('f1'))} | {fmt(c.get('f1'))} | {delta} | "
            f"{fmt(b.get('bleu1'))} | {fmt(c.get('bleu1'))} |"
        )
    adv = CATEGORY_LABELS[5]
    if adv in base and "refusal_rate" in base[adv]:
        rb = (refusal_bootstrap or {}).get(adv)
        delta = f"{rb['diff_mean']:+.4f} [{rb['ci_low']:+.4f}, {rb['ci_high']:+.4f}]" if rb else "--"
        lines.append("")
        lines.append(
            f"Adversarial, refusal reading (picked 'Not mentioned'): baseline "
            f"{fmt(base[adv]['refusal_rate'])}, candidate {fmt(cand[adv].get('refusal_rate'))}, "
            f"delta {delta}. The F1 row above is the paper's reading (scored against the trap text)."
        )
    errors = {arm: summary[arm]["overall"]["errors"] for arm in summary}
    if any(errors.values()):
        lines.append("")
        lines.append(f"Errors (scored 0): {errors}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# whole comparison
# ---------------------------------------------------------------------------


def run_test_eval(
    adapter: AMemGEPAAdapter,
    baseline_candidate: dict[str, str],
    candidate: dict[str, str] | None,
    instances: list[LoCoMoInstance],
    results_dir: Path,
    bootstrap_resamples: int = 10_000,
    meta: dict | None = None,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    arms = {BASELINE_ARM: baseline_candidate}
    if candidate is not None:
        arms[CANDIDATE_ARM] = candidate

    per_arm: dict[str, list[QuestionResult]] = {}
    for arm, prompts in arms.items():
        logger.info("=== arm %s: %d conversations, %d questions ===", arm, len(_by_conversation(instances)), len(instances))
        per_arm[arm] = evaluate_arm(adapter, arm, prompts, instances, results_dir)

    summary = {arm: summarize_arm(rs) for arm, rs in per_arm.items()}
    bootstrap = refusal_bootstrap = None
    if candidate is not None:
        bootstrap = paired_bootstrap(per_arm[BASELINE_ARM], per_arm[CANDIDATE_ARM], "f1", bootstrap_resamples)
        refusal_bootstrap = paired_bootstrap(
            per_arm[BASELINE_ARM], per_arm[CANDIDATE_ARM], "refusal_rate", bootstrap_resamples
        )
    table = format_comparison(summary, bootstrap or {}, refusal_bootstrap)

    report = {
        "meta": meta or {},
        "n_questions": len(instances),
        "conversations": sorted(_by_conversation(instances)),
        "summary": summary,
        "bootstrap_f1": bootstrap,
        "bootstrap_refusal": refusal_bootstrap,
        "adapter_stats": adapter.stats(),
        "results": {arm: [asdict(r) for r in rs] for arm, rs in per_arm.items()},
    }
    (results_dir / "test_eval.json").write_text(json.dumps(report, indent=2))
    (results_dir / "summary.md").write_text(table + "\n")
    return report
