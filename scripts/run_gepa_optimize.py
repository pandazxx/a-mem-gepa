#!/usr/bin/env python
"""Run a GEPA optimization over the paper-repro pipeline's four memory
prompts (docs/decisions/0014, 0015) -- all four round-robin by default, or
ONE of them with the other three frozen at the seed when --component is
given (docs/decisions/0019: one stage per prompt, note_construction first,
each stage seeded from the previous stage's winner via --seed-file).

Costs real API money and wall-clock (each memory build is a full
conversation replay -- ~45 min / ~$0.35 on GPT-4o-mini per
docs/experiments/002's calibration). Per CLAUDE.md's "Cost awareness", this
script REFUSES to start without --yes: run it once without --yes to see the
plan (models, budget, estimated builds), confirm with whoever pays, then
rerun with --yes.
"""

from __future__ import annotations

import json
import logging
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
    run_label: str = typer.Option(
        None,
        "--run-label",
        help="Results dir name; default: gepa_<timestamp>. Reuse a label to RESUME an interrupted "
        "run: GEPA state is reloaded from results/<label>/gepa_state.bin, finished builds come "
        "from the build cache, and an in-progress build resumes from its last turn checkpoint "
        "(docs/decisions/0017).",
    ),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Tiny sanity run on truncated conversations (gepa_smoke config section): validates the "
        "build -> cache -> QA -> reflection -> acceptance loop end to end. Plumbing only -- its "
        "numbers mean nothing (docs/decisions/0015).",
    ),
    component: str = typer.Option(
        None,
        "--component",
        help="Optimize ONLY this prompt (note_construction, evolution_decision, evolution_strengthen, "
        "evolution_update_neighbors); the other three are frozen byte-for-byte at the seed "
        "(docs/decisions/0019). Applies the gepa_single_component config overrides.",
    ),
    seed_candidate_dir: str = typer.Option(
        None,
        "--seed-candidate-dir",
        help="Directory with the 4 seed prompt files; default: gepa.seed_candidate_dir "
        "(the paper's prompts). E.g. a previous stage's results/<label>/best_candidate/.",
    ),
    seed_file: list[str] | None = typer.Option(  # noqa: B008 -- typer idiom, as the options above
        None,
        "--seed-file",
        help="component=path: replace one seed component with that file (repeatable). How a stage "
        "starts from the previous stage's committed winner, e.g. "
        "--seed-file note_construction=src/amem_gepa/prompts/note_construction.v1.txt",
    ),
    max_metric_calls: int = typer.Option(
        None, "--max-metric-calls", help="Override the config's metric-call budget for this run."
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
    from amem_gepa.gepa_stages import (
        COMPONENT_ORDER,
        FixedComponentSelector,
        changed_components,
        frozen_components,
        load_seed_candidate,
        parse_seed_override,
    )
    from amem_gepa.reflection_lm import ReflectionSettings, build_reflection_lm, reflection_stats
    from amem_gepa.repro_injection import (
        COMPONENTS,
        candidate_hash,
        load_baseline_candidate,
        validate_candidate,
    )

    cfg = load_config(config)
    gepa_cfg = dict(cfg["gepa"])
    if component:
        if component not in COMPONENTS:
            typer.echo(f"ERROR: --component must be one of {', '.join(COMPONENTS)}, got {component!r}")
            raise typer.Exit(code=1)
        # Single-component stages get their own budget (docs/decisions/0019);
        # smoke overrides still win below so `just gepa-smoke --component x`
        # stays a tiny plumbing check.
        gepa_cfg.update(cfg.get("gepa_single_component", {}))
    if smoke:
        gepa_cfg.update(cfg.get("gepa_smoke", {}))
    if max_metric_calls is not None:
        gepa_cfg["max_metric_calls"] = max_metric_calls
    backend = cfg["models"]["paper_repro_backend"]
    model = cfg["models"]["paper_repro_model"]
    reflection_lm = cfg["models"]["gepa_reflection_lm"]
    reflection = ReflectionSettings.from_config(reflection_lm, gepa_cfg.get("reflection"))
    # One knob for the task model's Ollama thinking mode, shared with `just
    # reproduce` and `just gepa-test-eval` (docs/decisions/0021).
    task_think = bool(cfg.get("paper_repro", {}).get("thinking", False))

    seed_dir = Path(seed_candidate_dir or gepa_cfg["seed_candidate_dir"])
    try:
        seed_overrides = dict(parse_seed_override(spec) for spec in (seed_file or []))
        seed_candidate = load_seed_candidate(seed_dir, seed_overrides)
    except (ValueError, OSError) as e:
        typer.echo(f"ERROR: could not assemble the seed candidate: {e}")
        raise typer.Exit(code=1)
    problems = validate_candidate(seed_candidate)
    if problems:
        typer.echo(f"Seed candidate invalid: {problems}")
        raise typer.Exit(code=1)
    baseline_candidate = load_baseline_candidate(Path(cfg["gepa"]["seed_candidate_dir"]))
    seed_differs_from_paper = changed_components(seed_candidate, baseline_candidate)
    frozen = frozen_components(seed_candidate, component) if component else {}

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

    default_label = (
        f"gepa_{'smoke_' if smoke else ''}{component + '_' if component else ''}"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    label = run_label or default_label
    run_dir = Path("results") / label

    train_convs = sorted({g.conversation_id for g in trainset})
    val_convs = sorted({i.conversation_id for i in valset})
    minibatch = gepa_cfg.get("reflection_minibatch_size", 2)

    typer.echo(f"GEPA run plan{' (SMOKE -- plumbing check, numbers are meaningless)' if smoke else ''}")
    typer.echo(f"  task backend/model:   {backend} / {model}")
    if backend == "ollama":
        typer.echo(
            f"  task thinking:        {'ON' if task_think else 'off'} (paper_repro.thinking, "
            f"docs/decisions/0016/0021{'; separate build cache' if task_think else ''})"
        )
    if smoke:
        typer.echo(f"  smoke truncation:     first {gepa_cfg.get('max_turns', 40)} turns per conversation")
    typer.echo(f"  reflection (coach):   {reflection.describe()}")
    if component:
        typer.echo(f"  optimizing:           {component} ONLY (docs/decisions/0019)")
        typer.echo(f"  frozen at the seed:   {', '.join(frozen)}")
        stage_idx = COMPONENT_ORDER.index(component)
        typer.echo(
            f"  stage:                {stage_idx + 1}/{len(COMPONENT_ORDER)} in the suggested order "
            f"{' -> '.join(COMPONENT_ORDER)}"
        )
    else:
        typer.echo(f"  components:           {', '.join(COMPONENTS)} (round-robin, all four mutable)")
    typer.echo(f"  seed candidate:       {seed_dir}")
    for name, path in sorted(seed_overrides.items()):
        typer.echo(f"    override:           {name} <- {path}")
    typer.echo(
        "  seed vs paper:        "
        + (
            f"differs on {', '.join(seed_differs_from_paper)} (a later stage)"
            if seed_differs_from_paper
            else "identical to the paper's prompts"
        )
    )
    typer.echo(f"  trainset:             {len(trainset)} groups over {len(train_convs)} conversations")
    typer.echo(f"  valset:               {len(valset)} questions over {len(val_convs)} conversations")
    typer.echo(f"  reflection minibatch: {minibatch} groups (= up to {minibatch} memory builds/iteration)")
    typer.echo(f"  max_metric_calls:     {gepa_cfg['max_metric_calls']}")
    typer.echo(
        f"  budget arithmetic:    each full val eval costs {len(valset)} metric calls "
        f"(+{len(val_convs)} builds, cached per candidate); each minibatch costs {minibatch}."
    )
    typer.echo(f"  adversarial scoring:  {gepa_cfg.get('adversarial_scoring', 'refusal')} (docs/decisions/0015)")
    resuming = (run_dir / "gepa_state.bin").exists()
    resume_note = "  (RESUMING -- gepa_state.bin found)" if resuming else ""
    typer.echo(f"  run dir:              {run_dir}{resume_note}")
    typer.echo(
        f"  build checkpoints:    every {gepa_cfg.get('build_checkpoint_every_turns', 20)} turns; "
        f"progress lines every {gepa_cfg.get('progress_log_every', 10)} turns/questions"
    )
    typer.echo(
        f"  interrupt/resume:     Ctrl-C (or `touch {run_dir}/gepa.stop` for a graceful stop), "
        f"then rerun with --run-label {label}"
    )

    if not yes:
        typer.echo("\nDry plan only. Confirm budget/models per CLAUDE.md, then rerun with --yes.")
        raise typer.Exit(code=0)

    # A resumed run must be the same stage: GEPA's state file carries the
    # candidate pool but not which component was being optimized or which
    # seed it grew from. Resuming under a different --component would freeze
    # the wrong three prompts and zero-score the pool's own candidates.
    stage_path = run_dir / "stage.json"
    stage_info = {"component": component, "seed_hash": candidate_hash(seed_candidate)}
    if stage_path.exists():
        prior = json.loads(stage_path.read_text())
        if prior != stage_info:
            typer.echo(
                f"ERROR: {run_dir} was started as {prior} but this invocation is {stage_info}. "
                "Resume with the same --component and seed, or pick a new --run-label."
            )
            raise typer.Exit(code=1)

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

    # Progress/ETA lines from the adapter (docs/decisions/0017) go to the
    # console AND results/<label>/amem_gepa.log, so a run started in the
    # background (CLAUDE.md's preference for long jobs) can be followed
    # with `tail -f`. GEPA's own engine log is run_log.txt in the same dir.
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_path.write_text(json.dumps(stage_info, indent=2) + "\n")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "amem_gepa.log")],
    )
    if resuming:
        logging.getLogger("amem_gepa.run_gepa_optimize").info("resuming GEPA run from %s", run_dir)

    adapter = AMemGEPAAdapter(
        backend=backend,
        model=model,
        retrieve_k=gepa_cfg.get("retrieve_k", 10),
        temperature_c5=gepa_cfg.get("temperature_c5", 0.5),
        cache_dir=Path(gepa_cfg.get("cache_dir", "results/gepa_cache")),
        adversarial_scoring=gepa_cfg.get("adversarial_scoring", "refusal"),
        checkpoint_every_turns=gepa_cfg.get("build_checkpoint_every_turns", 20),
        log_every=gepa_cfg.get("progress_log_every", 10),
        max_metric_calls=gepa_cfg["max_metric_calls"],
        stats_path=run_dir / "adapter_stats.json",
        frozen_components=frozen,
        task_think=task_think,
    )
    coach = build_reflection_lm(reflection)

    # Provenance for chaining stages and for the experiment doc's prompt
    # diff: the exact seed this run started from, next to its winner.
    seed_out = run_dir / "seed_candidate"
    seed_out.mkdir(parents=True, exist_ok=True)
    for name, text in seed_candidate.items():
        (seed_out / f"{name}.txt").write_text(text)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=coach,
        reflection_minibatch_size=minibatch,
        max_metric_calls=gepa_cfg["max_metric_calls"],
        # Coach spend cap (docs/decisions/0021); None = no cap. Ollama
        # reports $0 so the cap is inert there.
        max_reflection_cost=reflection.max_cost_usd,
        # One component per stage (docs/decisions/0019) or GEPA's default
        # round-robin over all four.
        module_selector=FixedComponentSelector(component) if component else "round_robin",
        run_dir=str(run_dir),
        display_progress_bar=True,
        raise_on_exception=False,
    )

    best_dir = run_dir / "best_candidate"
    best_dir.mkdir(parents=True, exist_ok=True)
    for name, text in result.best_candidate.items():
        (best_dir / f"{name}.txt").write_text(text)

    best_changed = changed_components(result.best_candidate, seed_candidate)
    frozen_intact = all(result.best_candidate.get(n) == t for n, t in frozen.items())
    if component and not frozen_intact:
        # Cannot happen with FixedComponentSelector + the adapter guard; if it
        # ever does, the stage's result is not a single-prompt result.
        typer.echo(f"ERROR: a frozen component changed in the best candidate: {best_changed}")

    summary = {
        "smoke": smoke,
        "resumed": resuming,
        "component": component,
        "frozen_components": sorted(frozen),
        "frozen_intact": frozen_intact,
        "seed_candidate_dir": str(seed_dir),
        "seed_overrides": {k: str(v) for k, v in sorted(seed_overrides.items())},
        "seed_differs_from_paper_on": seed_differs_from_paper,
        "best_changed_components": best_changed,
        "best_val_aggregate": result.val_aggregate_scores[result.best_idx],
        "num_candidates": len(result.candidates),
        # Whole-run totals, carried across interruptions via adapter_stats.json.
        **adapter.stats(),
        "config": config,
        "backend": backend,
        "model": model,
        "task_thinking": task_think,
        "reflection_lm": reflection_lm,
        "reflection": reflection.to_dict(),
        **reflection_stats(coach),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(
        f"\nBest candidate saved under {best_dir}/. To adopt it, copy each file to "
        "src/amem_gepa/prompts/<component>.v<N>.txt (never overwrite baseline_repro/, per CLAUDE.md) "
        "and write the docs/experiments/NNN-*.md entry, including Cost, in the same PR."
    )
    if component:
        nxt = COMPONENT_ORDER.index(component) + 1
        if nxt < len(COMPONENT_ORDER):
            typer.echo(
                f"Next stage (docs/decisions/0019), after the held-out check decides whether to keep "
                f"this winner:\n  just gepa-optimize-component {COMPONENT_ORDER[nxt]} "
                f"--seed-file {component}=src/amem_gepa/prompts/{component}.v<N>.txt --yes"
            )


if __name__ == "__main__":
    app()
