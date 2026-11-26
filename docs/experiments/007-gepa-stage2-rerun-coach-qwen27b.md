# 007: Stage 2 rerun — evolution decision only, coach changed to qwen3.8:27b

- Date: **not yet run**
- Config: `configs/stage2-coach-qwen27b.yaml` — identical to `configs/base.yaml`
  except `models.gepa_reflection_lm: ollama/qwen3.8:27b` pinned and
  `gepa.reflection.thinking: true` (commit hash: )
- GEPA budget: `max_metric_calls=800` (`gepa_single_component`),
  `reflection_minibatch_size=2`, task model `ollama/qwen3.5:9b` (thinking
  off, ADR 0016/0017), **reflection model `ollama/qwen3.8:27b`** (thinking:
  model default, i.e. on — same as 006's coach), `retrieve_k=10`,
  `adversarial_scoring=refusal`; coach thinking pinned on via
  `gepa.reflection.thinking: true` (ADR 0021) rather than left to the model
  default
- Component optimized: **`evolution_decision`**; frozen: `note_construction`,
  `evolution_strengthen`, `evolution_update_neighbors`
- Seed: identical to [006](006-gepa-stage2-evolution-decision-qwen3.5-9b.md)
  (`baseline_repro/` + `note_construction.v1.txt`)
- Command:
  `just gepa-optimize-component evolution_decision --seed-file note_construction=src/amem_gepa/prompts/note_construction.v1.txt --config configs/stage2-coach-qwen27b.yaml --run-label gepa_evolution_decision_coach-qwen27b --yes`

## What changed vs. the previous experiment

Exactly one thing: the coach. 006 found a `evolution_decision` rewrite
that is overall-significant on the paper's metric but a wash under the
training objective (refusal -8.5), leaked training-conversation entities
into the prompt, and buried its placeholders in one unlabelled sentence.
Same seed, same budget, same task model, same GEPA sampler seed (so the
same minibatch order); the only variable is whether a 27B coach writes a
better decision prompt than the 9B one. This is also the project's first
controlled data point on [ADR 0003](../decisions/0003-model-routing.md)'s
assumption that a stronger reflection model helps — one run per coach, so
a data point, not a result.

## Results

**Validation** (same run only — cross-run validation levels are not
comparable, see 006):

| | val aggregate |
|---|---|
| seed, as scored at the start of this run ("Base program full valset score" in `run_log.txt`) | TODO |
| this run's winner | TODO |
| for reference, 006's within-run pair | seed TODO / winner 0.3668 |

**Head to head, paired (the comparison this rerun exists for):** this
run's winner vs 006's winner, same test questions, one run.

```
just gepa-test-eval results/gepa_evolution_decision_coach-qwen27b/best_candidate \
    --baseline-dir results/gepa_evolution_decision_20261124_002304/best_candidate --yes
```

TODO: table.

**Stage gate: this run's winner vs stage 1's winner** (QA only once the
builds above exist):

```
just gepa-test-eval results/gepa_evolution_decision_coach-qwen27b/best_candidate \
    --baseline-dir results/gepa_note_construction_20261120_042519/best_candidate --yes
```

TODO: table, read as in 006 — overall interval *and* the refusal row.

## Pareto front summary

TODO: candidates / iterations / accepted; compare with 006's 6 / 11 / 5.

## Before/after prompt diff

TODO: commit as `src/amem_gepa/prompts/evolution_decision.v2.txt` whether or
not adopted. Specifically check the three things 006's prompt did: bias
toward evolving, training-entity leakage, placeholders outside a labelled
input block.

## Cost

TODO from `summary.json`: wall-clock, builds and min/build, QA, LLM calls,
metric calls. Coach: ~1 call per iteration; note the per-call time of the
27B with thinking on.

## Conclusion

TODO. Decision rule agreed in advance: seed stage 3 from this run's
winner only if, against stage 1's winner, the overall interval is above
zero **and** the refusal interval is not clearly below zero (proposal
criterion 2). Otherwise stage 3 seeds from stage 1's winner.
