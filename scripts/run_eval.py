#!/usr/bin/env python
"""Evaluate one prompt candidate (a GEPA output or a hand-written one) on a
split, and print/save the per-category + bootstrap-CI report described in
docs/experiments/README.md.

`--candidate` is a directory containing note_construction.txt and
evolution.txt (see src/amem_gepa/prompts/). run_baseline.py is this same
loop, pointed permanently at src/amem_gepa/prompts/baseline/.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from amem_gepa.run_eval_lib import run_and_report

app = typer.Typer()


@app.command()
def main(
    candidate: str = typer.Option(..., "--candidate", help="Dir with note_construction.txt + evolution.txt"),
    config: str = typer.Option(..., "--config"),
):
    load_dotenv()
    candidate_dir = Path(candidate)
    run_and_report(
        config_path=config,
        note_construction_path=candidate_dir / "note_construction.txt",
        evolution_path=candidate_dir / "evolution.txt",
        run_label=candidate_dir.name,
    )


if __name__ == "__main__":
    app()
