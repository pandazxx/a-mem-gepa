#!/usr/bin/env python
"""K-sweep for the paper-reproduction pipeline (docs/decisions/0013,
docs/experiments/001): finds the best `retrieve_k` for a fixed
backend/model, reusing memories already cached by a prior `just reproduce`
run under results/paper_repro/ (built fresh on a first call otherwise).
Mirrors external/agentic-memory-repro/run_k_sweep.sh's own K_VALUES,
adapted to the single-model-at-a-time, no-vLLM setup this project actually
uses (docs/decisions/0005).

Cost note -- NOT free: every k re-runs the full QA-answering step (2 LLM
calls/question) across all questions; it only skips memory-building. The
default sweep (k in {10,15,...,50}, 9 values) is 9x the QA-answering cost
of a single `just reproduce` run. Confirm this is worth it for your
backend/model before running for real (CLAUDE.md's cost-awareness
convention) -- resumable per-k under results/paper_repro/k_sweep/ if
interrupted partway.

Which k actually matters (read directly from the paper, Appendix A.5
Table 8, docs/decisions/0013): most models use k=10 for every category in
the paper's own headline numbers -- including Llama-3.2-1b, this
project's M2 model, so a sweep isn't expected to move that number. Only
GPT-4o-mini/GPT-4o are swept away from k=10 (k=40 for Multi Hop/Temporal/
Adversarial, k=50 for Open Domain/Single Hop). For those two models, the
cheapest way to reconstruct the paper's actual per-category number is
`--k-values 40,50` (2 values), not the full default sweep.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from amem_gepa.config import load_config
from amem_gepa.datasets.locomo import RAW_PATH, download_raw
from amem_gepa.paper_repro import (
    K_SWEEP_VALUES,
    PAPER_CATEGORY_K,
    format_k_sweep_summary,
    format_spliced_summary,
    run_k_sweep,
    splice_per_category_results,
)

RESULTS_DIR = Path("results/paper_repro")

app = typer.Typer()


def _try_print_spliced_summary(model: str, results_by_k: dict) -> None:
    """If this model has a known per-category k (PAPER_CATEGORY_K, e.g.
    GPT-4o-mini/GPT-4o via OpenRouter -- docs/decisions/0013 Table 8),
    reconstruct and print the paper-matching spliced comparison. Loads any
    needed k not already in `results_by_k` from a prior k-sweep's cached
    output on disk (results_dir/k_sweep/) rather than requiring every
    needed k to have been run in this exact invocation -- this is
    deliberately exact-match only on `model` (after stripping an
    "provider/" prefix, e.g. "openai/gpt-4o-mini" -> "gpt-4o-mini"), not
    fuzzy: Ollama's "llama3.2:1b" vs the paper's "llama3.2-1b" naming don't
    match here, and that's fine since Llama-3.2-1b uses a uniform k=10
    anyway (no splice needed). Silently does nothing if there's no match
    or a needed k hasn't been run/cached yet."""
    model_key = model.rsplit("/", 1)[-1].lower()
    category_k = PAPER_CATEGORY_K.get(model_key)
    if category_k is None:
        return

    combined = dict(results_by_k)
    missing = []
    for k in set(category_k.values()):
        if k in combined:
            continue
        cached_file = RESULTS_DIR / "k_sweep" / f"results_k{k}.json"
        if cached_file.exists():
            combined[k] = json.loads(cached_file.read_text())
        else:
            missing.append(k)

    if missing:
        print(
            f"\n[k-sweep] paper-matching splice for {model_key!r} needs k={sorted(set(category_k.values()))}, "
            f"but k={missing} hasn't been run yet -- skipping the spliced comparison for now."
        )
        return

    print(f"\nSpliced per-category comparison for {model_key!r} (paper's own k per category, docs/decisions/0013):")
    print(format_spliced_summary(splice_per_category_results(combined, category_k)))


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml", "--config"),
    k_values: str = typer.Option(
        ",".join(str(k) for k in K_SWEEP_VALUES),
        "--k-values",
        help="Comma-separated retrieve_k values to sweep",
    ),
    ratio: float = typer.Option(None, "--ratio", help="Fraction of the 10 conversations to evaluate; overrides config"),
):
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(config)
    backend = cfg["models"]["paper_repro_backend"]
    model = cfg["models"]["paper_repro_model"]
    repro_cfg = cfg["paper_repro"]
    r = ratio if ratio is not None else repro_cfg["ratio"]
    ks = tuple(int(k) for k in k_values.split(","))

    dataset_path = download_raw(RAW_PATH)

    print(
        f"[k-sweep] backend={backend!r} model={model!r} k_values={ks} ratio={r} "
        f"dataset={dataset_path}"
    )
    print(
        "[k-sweep] reuses memories cached under results/paper_repro/ "
        "(docs/decisions/0013) -- each k still re-runs QA-answering in full; "
        "resumable per-k under results/paper_repro/k_sweep/ if interrupted."
    )

    results_by_k = run_k_sweep(
        dataset_path=dataset_path,
        backend=backend,
        model=model,
        k_values=ks,
        ratio=r,
        temperature_c5=repro_cfg["temperature_c5"],
        results_dir=RESULTS_DIR,
    )

    print()
    print(format_k_sweep_summary(results_by_k))
    print(
        "\nCompare the best-k row above against the paper's Llama 3.2 1B / "
        "A-Mem row (docs/experiments/001) -- if it closes most of the gap, "
        "rerun `just reproduce --retrieve-k <best>` to make that the "
        "committed baseline."
    )
    _try_print_spliced_summary(model, results_by_k)


if __name__ == "__main__":
    app()
