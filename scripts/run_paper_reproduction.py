#!/usr/bin/env python
"""Faithful, end-to-end reproduction of the A-MEM paper's LoCoMo benchmark
(docs/decisions/0013) -- runs the authors' own reproduction code
(WujiangXu/AgenticMemory, vendored at external/agentic-memory-repro/), not
our own approximation. See src/amem_gepa/paper_repro.py for what's reused
unmodified vs. what's different from run_baseline.py's pipeline.

Full 10-conversation dataset by default (--ratio to sample fewer for a
quick check), retrieve_k=10 (their default -- the paper's headline numbers
used a per-model k-sweep, see docs/decisions/0013 on why we start here
instead). One model at a time, per docs/decisions/0005.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from amem_gepa.config import load_config
from amem_gepa.datasets.locomo import RAW_PATH, download_raw
from amem_gepa.paper_repro import format_summary, run_full_reproduction

app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml", "--config"),
    retrieve_k: int = typer.Option(None, "--retrieve-k", help="Overrides configs/base.yaml's paper_repro.retrieve_k"),
    ratio: float = typer.Option(None, "--ratio", help="Fraction of the 10 conversations to evaluate; overrides config"),
    output: str = typer.Option("results/paper_repro/results.json", "--output"),
):
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(config)
    backend = cfg["models"]["paper_repro_backend"]
    model = cfg["models"]["paper_repro_model"]
    repro_cfg = cfg["paper_repro"]
    k = retrieve_k if retrieve_k is not None else repro_cfg["retrieve_k"]
    r = ratio if ratio is not None else repro_cfg["ratio"]

    dataset_path = download_raw(RAW_PATH)

    print(
        f"[paper-repro] backend={backend!r} model={model!r} retrieve_k={k} "
        f"ratio={r} dataset={dataset_path} (full paper benchmark, docs/decisions/0013)"
    )
    print(
        "[paper-repro] resumable at the conversation level: memories for an "
        "already-processed conversation are cached under results/paper_repro/ "
        "and reused on rerun, no re-prompting the LLM for them."
    )

    results = run_full_reproduction(
        dataset_path=dataset_path,
        backend=backend,
        model=model,
        retrieve_k=k,
        ratio=r,
        temperature_c5=repro_cfg["temperature_c5"],
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}\n")
    print(format_summary(results))
    print(
        "\nCompare against the paper's Llama 3.2 1B / A-Mem row (or whichever "
        "model/row you ran) -- see the benchmark table shared earlier in this project."
    )


if __name__ == "__main__":
    app()
