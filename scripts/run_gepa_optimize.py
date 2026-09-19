#!/usr/bin/env python
"""Run a GEPA optimization over the paper-repro pipeline's four memory
prompts (docs/decisions/0014, 0015).

Costs real API money and wall-clock (each memory build is a full
conversation replay -- ~45 min / ~$0.35 on GPT-4o-mini per
docs/experiments/002's calibration). Per CLAUDE.md's "Cost awareness", this
script REFUSES to start without --yes: run it once without --yes to see the
plan (models, budget, estimated builds), confirm with whoever pays, then
rerun with --yes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv

app = typer.Typer()


def _val_manifest_entry(inst) -> dict:
    return {"conversation_id": inst.conversation_id, "question": inst.question, "category": inst.category}


@app.command()
def main(
    config: str = typer.Option(..., "--config"),
    yes: bool = typer.Option(False, "--yes", help="Actually start the run (otherwise: print the plan and exit)."),
    run_label: str = typer.Option(None, "--run-label", help="Results dir name; default: gepa_<timestamp>."),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Tiny sanity run on truncated conversations (gepa_smoke config section): validates the "
        "build -> cache -> QA -> reflection -> acceptance loop end to end. Plumbing only -- its "
        "numbers mean nothing (docs/decisions/0015).",
    ),
):
    load_dotenv()

    from amem_gepa.config import load_config
    from amem_gepa.datasets.locomo import (
        build_question_groups,
        build_val_subset,
        load_split,
        truncate_instances,
    )
    from amem_gepa.gepa_adapter import AMemGEPAAdapter
    from amem_gepa.repro_injection import COMPONENTS, load_baseline_candidate, validate_candidate

    cfg = load_config(config)
    gepa_cfg = dict(cfg["gepa"])
    if smoke:
        gepa_cfg.update(cfg.get("gepa_smoke", {}))
    backend = cfg["models"]["paper_repro_backend"]
    model = cfg["models"]["paper_repro_model"]
    reflection_lm = cfg["models"]["gepa_reflection_lm"]

    seed_candidate = load_baseline_candidate(Path(gepa_cfg["seed_candidate_dir"]))
    problems = validate_candidate(seed_candidate)
    if problems:
        typer.echo(f"Seed candidate invalid: {problems}")
        raise typer.Exit(code=1)

    train_instances = load_split("train")
    val_instances = load_split("val")
    if smoke:
        max_turns = gepa_cfg.get("max_turns", 40)
        train_instances = truncate_instances(train_instances, max_turns)
        val_instances = truncate_instances(val_instances, max_turns)
        keep = gepa_cfg.get("train_conversations", 2)
        kept_ids = sorted({i.conversation_id for i in train_instances})[:keep]
        train_instances = [i for i in train_instances if i.conversation_id in kept_ids]

    trainset = build_question_groups(
        train_instances, per_category=gepa_cfg["train_questions_per_category"]
    )
    valset = build_val_subset(
        val_instances,
        per_category_per_conversation=gepa_cfg["val_questions_per_category_per_conversation"],
    )
    if not trainset or not valset:
        typer.echo("ERROR: empty trainset or valset (smoke truncation too aggressive?)")
        raise typer.Exit(code=1)

    # The val subset is derived deterministically, but commit it as a
    # manifest anyway (same reasoning as configs/locomo_split.json): if the
    # dataset file ever changes upstream, the subset should change via a
    # reviewed diff, not silently. Smoke runs skip this entirely -- their
    # truncated subset is not the real one and must not touch the manifest.
    if not smoke:
        manifest_path = Path(gepa_cfg.get("val_subset_manifest", "configs/gepa_val_subset.json"))
        manifest = [_val_manifest_entry(inst) for inst in valset]
        if manifest_path.exists():
            committed = json.loads(manifest_path.read_text())
            if committed != manifest:
                typer.echo(
                    f"ERROR: derived val subset differs from committed {manifest_path} "
                    "(dataset drift or a parameter change). Regenerate the manifest deliberately "
                    "(delete it and rerun) and review the diff before optimizing."
                )
                raise typer.Exit(code=1)
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            typer.echo(f"Wrote val-subset manifest to {manifest_path} -- commit it with this run's PR.")

    default_label = f"gepa_{'smoke_' if smoke else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    label = run_label or default_label
    run_dir = Path("results") / label

    train_convs = sorted({g.conversation_id for g in trainset})
    val_convs = sorted({i.conversation_id for i in valset})
    minibatch = gepa_cfg.get("reflection_minibatch_size", 2)

    typer.echo(f"GEPA run plan{' (SMOKE -- plumbing check, numbers are meaningless)' if smoke else ''}")
    typer.echo(f"  task backend/model:   {backend} / {model}")
    if smoke:
        typer.echo(f"  smoke truncation:     first {gepa_cfg.get('max_turns', 40)} turns per conversation")
    typer.echo(f"  reflection_lm:        {reflection_lm}")
    typer.echo(f"  components:           {', '.join(COMPONENTS)}")
    typer.echo(f"  trainset:             {len(trainset)} groups over {len(train_convs)} conversations")
    typer.echo(f"  valset:               {len(valset)} questions over {len(val_convs)} conversations")
    typer.echo(f"  reflection minibatch: {minibatch} groups (= up to {minibatch} memory builds/iteration)")
    typer.echo(f"  max_metric_calls:     {gepa_cfg['max_metric_calls']}")
    typer.echo(
        f"  budget arithmetic:    each full val eval costs {len(valset)} metric calls "
        f"(+{len(val_convs)} builds, cached per candidate); each minibatch costs {minibatch}."
    )
    typer.echo(f"  adversarial scoring:  {gepa_cfg.get('adversarial_scoring', 'refusal')} (docs/decisions/0015)")
    typer.echo(f"  run dir:              {run_dir}")

    if not yes:
        typer.echo("\nDry plan only. Confirm budget/models per CLAUDE.md, then rerun with --yes.")
        raise typer.Exit(code=0)

    # Preflight: one tiny completion through the exact controller stack the
    # run will use, so a wrong OLLAMA_HOST / missing model fails in seconds
    # with a clear error instead of an hour into the first build. (Two
    # same-purpose env vars exist -- OLLAMA_HOST for this native-ollama
    # path, OLLAMA_API_BASE for the LiteLLM-routed pipelines; see
    # docs/decisions/0013's dual-variable warning.)
    from amem_gepa.paper_repro import ensure_repro_repo_importable

    ensure_repro_repo_importable()
    from memory_layer_robust import RobustLLMController

    typer.echo(f"Preflight: pinging {backend}/{model} ...")
    RobustLLMController(backend=backend, model=model, check_connection=True)
    typer.echo("Preflight OK.")

    import gepa

    adapter = AMemGEPAAdapter(
        backend=backend,
        model=model,
        retrieve_k=gepa_cfg.get("retrieve_k", 10),
        temperature_c5=gepa_cfg.get("temperature_c5", 0.5),
        cache_dir=Path(gepa_cfg.get("cache_dir", "results/gepa_cache")),
        adversarial_scoring=gepa_cfg.get("adversarial_scoring", "refusal"),
    )

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=minibatch,
        max_metric_calls=gepa_cfg["max_metric_calls"],
        run_dir=str(run_dir),
        display_progress_bar=True,
        raise_on_exception=False,
    )

    best_dir = run_dir / "best_candidate"
    best_dir.mkdir(parents=True, exist_ok=True)
    for component, text in result.best_candidate.items():
        (best_dir / f"{component}.txt").write_text(text)

    summary = {
        "smoke": smoke,
        "best_val_aggregate": result.val_aggregate_scores[result.best_idx],
        "num_candidates": len(result.candidates),
        "builds_performed": adapter.builds_performed,
        "build_cache_hits": adapter.build_cache_hits,
        "llm_calls": adapter.llm_calls,
        "config": config,
        "backend": backend,
        "model": model,
        "reflection_lm": reflection_lm,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(
        f"\nBest candidate saved under {best_dir}/. To adopt it, copy each file to "
        "src/amem_gepa/prompts/<component>.v<N>.txt (never overwrite baseline_repro/, per CLAUDE.md) "
        "and write the docs/experiments/NNN-*.md entry, including Cost, in the same PR."
    )


if __name__ == "__main__":
    app()
