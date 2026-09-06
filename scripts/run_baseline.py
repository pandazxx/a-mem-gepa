#!/usr/bin/env python
"""Evaluate A-MEM with the original, unmodified paper prompts on a split.

Thin wrapper around run_eval.py's loop, pointed at
src/amem_gepa/prompts/baseline/ -- see docs/decisions/0002 on why that
baseline is never hand-edited.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from amem_gepa.run_eval_lib import run_and_report

app = typer.Typer()

BASELINE_DIR = Path("src/amem_gepa/prompts/baseline")


@app.command()
def main(config: str = typer.Option(..., "--config")):
    load_dotenv()
    run_and_report(
        config_path=config,
        note_construction_path=BASELINE_DIR / "note_construction.txt",
        evolution_path=BASELINE_DIR / "evolution.txt",
        run_label="baseline",
    )


if __name__ == "__main__":
    app()
