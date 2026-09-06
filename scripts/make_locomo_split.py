#!/usr/bin/env python
"""Generate configs/locomo_split.json (docs/decisions/0004): 5/2/3
train/val/test conversations, chosen by exhaustive search to minimize
per-category distribution deviation from the whole dataset. Written once and
committed -- not re-randomized on every run (the search is deterministic
anyway, so re-running reproduces the same manifest as long as the upstream
dataset file hasn't changed).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from amem_gepa.datasets.locomo import load_raw_conversations, make_split

app = typer.Typer()


@app.command()
def main(out: str = typer.Option(..., "--out")):
    conversations = load_raw_conversations()
    split = make_split(conversations)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2) + "\n")
    print(f"Wrote {out_path}: " + ", ".join(f"{k}={len(v)}" for k, v in split.items()))


if __name__ == "__main__":
    app()
