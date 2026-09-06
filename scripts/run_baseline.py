#!/usr/bin/env python
"""Evaluate A-MEM with the original, unmodified paper prompts on the test split.

Not implemented yet -- milestone-2 work, once datasets/locomo.py and
metrics.py exist. Wired up now so `just baseline` has a stable entry point.
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

app = typer.Typer()


@app.command()
def main(config: str = typer.Option(..., "--config")):
    load_dotenv()
    raise NotImplementedError("milestone-2: baseline eval loop")


if __name__ == "__main__":
    app()
