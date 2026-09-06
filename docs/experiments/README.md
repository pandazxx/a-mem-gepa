# Experiment log convention

One file per run: `docs/experiments/NNN-short-name.md`, numbered sequentially.
Write it right after the run finishes, while the config and results are
still in `results/` — don't batch these up.

Template:

```markdown
# NNN: Short name

- Date:
- Config: configs/<file>.yaml (commit hash: )
- GEPA budget: max_metric_calls=, task_lm=, reflection_lm=

## What changed vs. the previous experiment

## Results

Aggregate (test set, with bootstrap CI):

Per-category breakdown:

## Pareto front summary

(How many candidates survived, what trade-offs they represent.)

## Before/after prompt diff

## Cost

Tokens / $ spent on this run.

## Conclusion

Did it meet the success criteria in docs/00-proposal.md? What's next?
```

Raw logs/outputs for a run live in `results/<NNN>/` (gitignored) — this doc is
the durable, committed summary of what happened.
