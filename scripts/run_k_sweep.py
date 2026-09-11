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
"""

from __future__ import annotations

import logging

import typer
from dotenv import load_dotenv

from amem_gepa.config import load_config
from amem_gepa.datasets.locomo import RAW_PATH, download_raw
from amem_gepa.paper_repro import K_SWEEP_VALUES, format_k_sweep_summary, run_k_sweep

app = typer.Typer()


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
    )

    print()
    print(format_k_sweep_summary(results_by_k))
    print(
        "\nCompare the best-k row above against the paper's Llama 3.2 1B / "
        "A-Mem row (docs/experiments/001) -- if it closes most of the gap, "
        "rerun `just reproduce --retrieve-k <best>` to make that the "
        "committed baseline."
    )


if __name__ == "__main__":
    app()
