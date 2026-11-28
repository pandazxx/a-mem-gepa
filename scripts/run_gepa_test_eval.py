#!/usr/bin/env python
"""Held-out test comparison: paper baseline prompts vs. a GEPA candidate, on
the paper-reproduction pipeline (docs/decisions/0018; library:
src/amem_gepa/gepa_test_eval.py).

Costs real wall-clock: one memory build per (arm, test conversation) -- 6
builds for baseline + candidate on the 3 test conversations, ~1.5 h each on a
local 9B model per the first real run -- plus every test question answered
twice. Per CLAUDE.md, run once WITHOUT --yes to see the plan, then rerun with
--yes. Resumable: rerun with the same --run-label.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option(..., "--config"),
    candidate: str = typer.Option(
        None,
        "--candidate",
        help="Directory holding the 4 candidate prompt files (e.g. results/<run>/best_candidate). "
        "Omit to score the baseline arm only.",
    ),
    split: str = typer.Option("test", "--split", help="Which split's conversations to evaluate (default: test)."),
    run_label: str = typer.Option(None, "--run-label", help="Results dir name; default: test_eval_<candidate dir name>. Reuse to resume."),
    retrieve_k: int = typer.Option(None, "--retrieve-k", help="Top-k at answer time for BOTH arms; default: gepa.retrieve_k from the config (what training used)."),
    yes: bool = typer.Option(False, "--yes", help="Actually run (otherwise: print the plan and exit)."),
    proxy: str = typer.Option(
        None,
        "--proxy",
        help="'evidence' or 'competitors': score the candidate arm on the baseline-built memory with only "
        "the question-relevant notes re-constructed (docs/decisions/0020). Candidate must differ from the "
        "baseline in note_construction only. Refused on the test split -- this is the calibration / "
        "surrogate path, never a reported number.",
    ),
):
    load_dotenv()

    from amem_gepa.config import load_config
    from amem_gepa.datasets.locomo import load_split
    from amem_gepa.gepa_adapter import AMemGEPAAdapter
    from amem_gepa.gepa_test_eval import run_test_eval
    from amem_gepa.repro_injection import load_baseline_candidate, validate_candidate

    cfg = load_config(config)
    gepa_cfg = cfg["gepa"]
    backend = cfg["models"]["paper_repro_backend"]
    model = cfg["models"]["paper_repro_model"]
    k = retrieve_k if retrieve_k is not None else gepa_cfg.get("retrieve_k", 10)
    # Proxy settings live in the single-component section (docs/decisions/0020).
    stage_cfg = cfg.get("gepa_single_component", {})
    competitors_k = stage_cfg.get("proxy_competitors_k", gepa_cfg.get("proxy_competitors_k", 10))
    resamples = cfg.get("evaluation", {}).get("bootstrap_resamples", 10_000)

    baseline = load_baseline_candidate(Path(gepa_cfg["seed_candidate_dir"]))
    cand = None
    if candidate:
        cand = load_baseline_candidate(Path(candidate))
        problems = validate_candidate(cand)
        if problems:
            typer.echo(f"Candidate invalid: {problems}")
            raise typer.Exit(code=1)
        if cand == baseline:
            typer.echo("Candidate is byte-identical to the baseline -- nothing to compare.")
            raise typer.Exit(code=1)

    proxy_mode = proxy or "none"
    if proxy_mode != "none":
        from amem_gepa.gepa_stages import changed_components

        if proxy_mode not in ("evidence", "competitors"):
            typer.echo(f"ERROR: --proxy must be 'evidence' or 'competitors', got {proxy_mode!r}")
            raise typer.Exit(code=1)
        if split == "test":
            typer.echo("ERROR: --proxy is a surrogate for calibration/training; the test split is full builds only (docs/decisions/0018, 0020).")
            raise typer.Exit(code=1)
        if cand is None or changed_components(cand, baseline) != ["note_construction"]:
            typer.echo("ERROR: --proxy requires a candidate that differs from the baseline in note_construction only.")
            raise typer.Exit(code=1)

    instances = load_split(split)
    convs = sorted({i.conversation_id for i in instances})
    label = run_label or (
        f"test_eval_{Path(candidate).parent.name + '_' if candidate else ''}{split}"
        + (f"_proxy_{proxy_mode}" if proxy_mode != "none" else "")
    )
    run_dir = Path("results") / label
    arms = 2 if cand is not None else 1
    resuming = any(run_dir.glob("qa_progress_*.jsonl"))

    typer.echo("Held-out comparison plan (docs/decisions/0018)")
    typer.echo(f"  backend/model:   {backend} / {model}")
    typer.echo(f"  split:           {split} -> {len(convs)} conversations {convs}, {len(instances)} questions")
    typer.echo(f"  arms:            baseline ({gepa_cfg['seed_candidate_dir']})" + (f" vs candidate ({candidate})" if cand else " only"))
    typer.echo(f"  retrieve_k:      {k} (same for both arms)")
    if proxy_mode != "none":
        typer.echo(
            f"  evaluation:      PROXY '{proxy_mode}' for the candidate arm"
            + (f", top-{competitors_k} competitors per question" if proxy_mode == "competitors" else "")
            + " (surrogate; baseline arm = cached full build)"
        )
    typer.echo(f"  cost:            up to {arms * len(convs)} memory builds (fewer if cached) + {arms * len(instances)} questions")
    typer.echo(f"  bootstrap:       {resamples} paired resamples")
    typer.echo(f"  run dir:         {run_dir}{'  (RESUMING -- progress logs found)' if resuming else ''}")
    if not yes:
        typer.echo("\nDry plan only. Rerun with --yes to start.")
        raise typer.Exit(code=0)

    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "test_eval.log")],
    )

    from amem_gepa.paper_repro import ensure_repro_repo_importable

    ensure_repro_repo_importable()
    from memory_layer_robust import RobustLLMController

    typer.echo(f"Preflight: pinging {backend}/{model} ...")
    RobustLLMController(backend=backend, model=model, check_connection=True)
    typer.echo("Preflight OK.")

    adapter = AMemGEPAAdapter(
        backend=backend,
        model=model,
        retrieve_k=k,
        temperature_c5=gepa_cfg.get("temperature_c5", 0.5),
        cache_dir=Path(gepa_cfg.get("cache_dir", "results/gepa_cache")),
        checkpoint_every_turns=gepa_cfg.get("build_checkpoint_every_turns", 20),
        log_every=gepa_cfg.get("progress_log_every", 10),
        stats_path=run_dir / "adapter_stats.json",
        **(
            {
                "frozen_components": {c: t for c, t in baseline.items() if c != "note_construction"},
                "seed_candidate": baseline,
                "proxy_mode": proxy_mode,
                "proxy_competitors_k": competitors_k,
            }
            if proxy_mode != "none"
            else {}
        ),
    )
    report = run_test_eval(
        adapter,
        baseline,
        cand,
        instances,
        run_dir,
        bootstrap_resamples=resamples,
        meta={"config": config, "backend": backend, "model": model, "retrieve_k": k, "split": split,
              "candidate_dir": candidate, "baseline_dir": gepa_cfg["seed_candidate_dir"],
              "proxy_mode": proxy_mode},
    )
    typer.echo("\n" + (run_dir / "summary.md").read_text())
    typer.echo(json.dumps(report["adapter_stats"], indent=2))
    typer.echo(
        f"\nWritten: {run_dir}/test_eval.json, {run_dir}/summary.md. Paste summary.md into the run's "
        "docs/experiments/NNN-*.md (Results) with adapter_stats under Cost."
    )


if __name__ == "__main__":
    app()
