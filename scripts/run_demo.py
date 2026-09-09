#!/usr/bin/env python
"""Tiny, fast end-to-end sanity demo of the A-MEM pipeline (docs/decisions/0008):
one conversation truncated to a handful of turns, a handful of answerable
questions. Not a reproduction -- verifies replay -> retrieve -> answer ->
score runs correctly and quickly, before committing to the full
multi-conversation, multi-hundred-question milestone-2 reproduction run
(which took multiple days against Llama 3.2:1b).

Prints every question/prediction/score, not just a summary, since the point
is a human sanity-checking whether the answers look reasonable. Traces the
internals by default (docs/decisions/0011): note construction, evolution
decisions, retrieval, and the QA-answer call, so the whole pipeline is
visible turn by turn -- `just baseline`/`just eval` never enable this, so
a real reproduction run's output stays clean. Pass --no-trace for the
quieter Q/A-only view.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from amem_gepa.config import load_config
from amem_gepa.datasets.locomo import CATEGORY_LABELS, load_demo_sample
from amem_gepa.evaluate import evaluate_candidate

app = typer.Typer()

BASELINE_DIR = Path("src/amem_gepa/prompts/baseline")


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml", "--config"),
    max_turns: int = typer.Option(15, "--max-turns"),
    max_questions: int = typer.Option(5, "--max-questions"),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Print note construction/evolution/retrieval/answer details"),
):
    load_dotenv()
    cfg = load_config(config)
    instances = load_demo_sample(max_turns=max_turns, max_questions=max_questions)

    if not instances:
        print("No answerable questions found in the first max_turns turns -- try raising --max-turns.")
        raise typer.Exit(1)

    print(
        f"[demo] {len(instances)} questions from conversation "
        f"{instances[0].conversation_id!r}, first {max_turns} turns, "
        f"model={cfg['models']['amem_llm_model']!r}"
    )
    if trace:
        print(
            "[demo] tracing enabled -- this re-runs live even if you've run "
            "`just demo` before, ignoring any cached predictions, so you can "
            "watch it happen (pass --no-trace to skip this and use the cache)"
        )

    result = evaluate_candidate(
        instances,
        note_construction_prompt=(BASELINE_DIR / "note_construction.txt").read_text(),
        evolution_prompt=(BASELINE_DIR / "evolution.txt").read_text(),
        qa_prompt_template=Path(cfg["evaluation"]["qa_prompt"]).read_text(),
        llm_model=cfg["models"]["amem_llm_model"],
        llm_api_base=cfg["models"].get("amem_llm_api_base") or None,
        embedding_model=cfg["models"]["amem_embedding_model"],
        k=cfg["evaluation"]["retrieval_k"],
        n_bootstrap_resamples=200,
        run_label="demo",
        split="demo",
        trace=trace,
    )

    for r in result.instance_results:
        if r.category == 5:
            # Correct behavior here is to NOT match this -- it's a trap
            # answer LoCoMo grounds in something real Jon/Gina/etc. said
            # about a *different* topic, not a gold answer to reproduce.
            label = "trap (correct = does NOT match)"
            reference = r.adversarial_answer
        else:
            label = "expected"
            reference = r.gold_answer
        print(
            f"\nQ ({CATEGORY_LABELS[r.category]}): {r.question}\n"
            f"  {label}: {reference!r}\n"
            f"  predicted: {r.prediction!r}\n"
            f"  score: {r.score:.2f}"
        )

    print("\n[demo] summary:")
    for name, summary in result.summaries.items():
        print(f"  {name:12s} mean={summary.mean:.3f}  n={summary.n}")

    print(
        "\nIf these look sane, run `just baseline` for the full reproduction "
        "(docs/decisions/0005) -- it's resumable now (docs/decisions/0007), "
        "but still takes a long time against a local model."
    )


if __name__ == "__main__":
    app()
