#!/usr/bin/env python
"""Evaluate one prompt candidate (a GEPA output or a hand-written one) on the
held-out test split, and print/save the per-category + bootstrap-CI report
described in docs/experiments/README.md.

Not implemented yet -- milestone-2/4 work, shares its eval loop with
run_baseline.py (which is just this script pointed at the baseline prompts).
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

app = typer.Typer()


@app.command()
def main(
    candidate: str = typer.Option(..., "--candidate", help="Path to a candidate prompt dir"),
    config: str = typer.Option(..., "--config"),
):
    load_dotenv()
    raise NotImplementedError("milestone-2: shared eval loop with run_baseline.py")


if __name__ == "__main__":
    app()
