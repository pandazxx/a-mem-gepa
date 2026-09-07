"""Shared body of scripts/run_baseline.py and scripts/run_eval.py: load a
config + a prompt candidate, run evaluate.evaluate_candidate over the
configured split, print a per-category report, and save the full result as
JSON under results/ (gitignored -- docs/experiments/NNN-*.md is the
committed, durable summary, per CLAUDE.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from amem_gepa.config import load_config
from amem_gepa.datasets.locomo import load_split
from amem_gepa.evaluate import evaluate_candidate


def run_and_report(
    config_path: str,
    note_construction_path: Path,
    evolution_path: Path,
    run_label: str,
) -> dict:
    config = load_config(config_path)
    note_construction_prompt = Path(note_construction_path).read_text()
    evolution_prompt = Path(evolution_path).read_text()
    qa_prompt = Path(config["evaluation"]["qa_prompt"]).read_text()

    split = config["evaluation"]["split"]
    instances = load_split(split, manifest_path=Path(config["dataset"]["split_manifest"]))

    print(
        f"[{run_label}] evaluating {len(instances)} QA pairs from split={split!r} "
        f"using model={config['models']['amem_llm_model']!r} (resumable -- rerun the same "
        f"command after an interruption to continue instead of restarting)"
    )

    result = evaluate_candidate(
        instances,
        note_construction_prompt=note_construction_prompt,
        evolution_prompt=evolution_prompt,
        qa_prompt_template=qa_prompt,
        llm_model=config["models"]["amem_llm_model"],
        llm_api_base=config["models"].get("amem_llm_api_base") or None,
        embedding_model=config["models"]["amem_embedding_model"],
        k=config["evaluation"]["retrieval_k"],
        n_bootstrap_resamples=config["evaluation"]["bootstrap_resamples"],
        run_label=run_label,
        split=split,
    )

    print(f"\n{run_label} results (split={split}):")
    for name, summary in result.summaries.items():
        print(
            f"  {name:12s} mean={summary.mean:.3f}  "
            f"95% CI=[{summary.ci_low:.3f}, {summary.ci_high:.3f}]  n={summary.n}"
        )

    out_dir = Path("results") / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.json"
    out_path.write_text(
        json.dumps(
            {
                "run_label": run_label,
                "split": split,
                "model": config["models"]["amem_llm_model"],
                "summaries": {k: asdict(v) for k, v in result.summaries.items()},
                "instance_results": [asdict(r) for r in result.instance_results],
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")
    return result.summaries
