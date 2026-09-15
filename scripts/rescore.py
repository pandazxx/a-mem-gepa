#!/usr/bin/env python
"""Recompute summaries for an already-completed run's results/<run_label>/<split>.json
using the current metrics.py, without redoing any LLM calls. Exists because
evaluate_candidate's per-question cache stores predictions (expensive) and
used to trust their stored score (cheap, but can go stale when metrics.py
changes -- see the tokenization fix in metrics.py's git history). The cache
itself is now always rescored fresh on every run/resume; this script is for
rescoring a JSON report that was already written out, without re-running
`just baseline` at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from amem_gepa.metrics import score_answer, score_batch

app = typer.Typer()


@app.command()
def main(results_path: str = typer.Argument(..., help="e.g. results/baseline/test.json")):
    path = Path(results_path)
    data = json.loads(path.read_text())
    instance_results = data["instance_results"]

    summaries = score_batch(
        [r["prediction"] for r in instance_results],
        [r["category"] for r in instance_results],
        [r["gold_answer"] for r in instance_results],
        [r["adversarial_answer"] for r in instance_results],
    )

    print(f"Rescored {len(instance_results)} instances from {path} with the current metrics.py:\n")
    for name, summary in summaries.items():
        print(
            f"  {name:12s} mean={summary.mean:.3f}  "
            f"95% CI=[{summary.ci_low:.3f}, {summary.ci_high:.3f}]  n={summary.n}"
        )

    for r in instance_results:
        r["score"] = score_answer(
            r["prediction"], r["category"], gold_answer=r["gold_answer"], adversarial_answer=r["adversarial_answer"]
        )

    data["summaries"] = {k: asdict(v) for k, v in summaries.items()}
    data["instance_results"] = instance_results
    path.write_text(json.dumps(data, indent=2))
    print(f"\nUpdated {path} in place.")


if __name__ == "__main__":
    app()
