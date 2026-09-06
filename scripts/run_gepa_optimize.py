#!/usr/bin/env python
"""Run a GEPA optimization job over A-MEM's note-construction + evolution
prompts (docs/decisions/0002).

Costs real API calls -- see CLAUDE.md's "Cost awareness" section. Confirm
max_metric_calls and task_lm/reflection_lm with whoever's running this before
it's actually wired up.

Not implemented yet -- milestone-3 work, once gepa_adapter.py's evaluate()
and make_reflective_dataset() are filled in.
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

app = typer.Typer()


@app.command()
def main(config: str = typer.Option(..., "--config")):
    load_dotenv()
    raise NotImplementedError("milestone-3: gepa.optimize() call using AMemGEPAAdapter")


if __name__ == "__main__":
    app()
