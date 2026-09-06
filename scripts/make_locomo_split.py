#!/usr/bin/env python
"""Generate configs/locomo_split.json (docs/decisions/0004): 5/2/3
train/val/test conversations, balanced by QA category mix, written once and
committed -- not re-randomized on every run.

Not implemented yet -- milestone-1 work, blocked on fetching the actual
LoCoMo dataset and inspecting its conversation ID / category schema first.
"""

from __future__ import annotations

import typer

app = typer.Typer()


@app.command()
def main(out: str = typer.Option(..., "--out")):
    raise NotImplementedError("milestone-1: fetch LoCoMo, call datasets.locomo.make_split()")


if __name__ == "__main__":
    app()
