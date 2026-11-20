# 0017: GEPA on a local Ollama model — thinking-mode parity, progress/ETA, mid-build resume

## Status

Accepted (2026-11-08). Revisits the "accept coarse per-conversation resume
until it hurts" note in [0013](0013-paper-reproduction-pipeline.md) for GEPA
builds specifically; extends [0016](0016-disable-ollama-thinking-mode.md)'s
`think=False` fix to GEPA rollouts.

## Context

The user reproduced the paper pipeline with a local `qwen3.5:9b`
([experiments/003](../experiments/003-qwen3.5-9b-exploration.md)) and
intends to use the same model for the GEPA evaluation and, if viable, for
GEPA training (`task_lm`). Three things followed from checking the GEPA
path (`gepa_adapter.py`, `scripts/run_gepa_optimize.py`, PR #5) against
what the reproduction runs taught:

1. **Thinking mode was only disabled for `just reproduce`.** 0016's
   `_patch_ollama_chat()` (which forces `think=False` on every Ollama
   call) is installed inside `paper_repro.run_full_reproduction` only. The
   GEPA adapter constructs `RobustAdvancedMemAgent` itself and never
   installed it -- so a GEPA run on `qwen3.5:9b` would have reproduced the
   ~100x per-call slowdown 003 spent weeks diagnosing, *and* would have
   compared GEPA candidates (thinking on) against a baseline built with
   thinking off: the same model under different generation settings.
2. **No progress or ETA.** The run script enabled GEPA's tqdm bar over
   metric calls and nothing else: no `logging.basicConfig`, so the
   adapter's own `logger.info` lines (cache hits) were silent, and the
   build loop -- hundreds of LLM calls, ~1h per conversation on a local
   model -- printed nothing. 003's diagnosis only became possible after
   the reproduction got exactly this kind of per-call visibility.
3. **Resume was partial.** GEPA saves `gepa_state.bin` at the start of
   every iteration and reloads it when `run_dir` already exists, and
   finished builds are cached per (candidate, conversation) -- so
   rerunning with the same `--run-label` already resumed at the iteration
   level. But the default label is a timestamp (so a bare rerun starts a
   new run), the adapter's run totals reset on resume, and an
   *in-progress* build had no checkpoint at all: a kill at turn 180 of 200
   replayed all 200. 0013 accepted that coarseness for the reproduction
   "until it becomes actual observed pain"; for GEPA, where the build is
   the budget currency ([0015](0015-gepa-training-method.md)) and the
   model is slow and local, it is.

## Decision

- **Same model for training is supported and is the default
  configuration.** GEPA's task model is already whatever
  `PAPER_REPRO_BACKEND`/`PAPER_REPRO_MODEL` name
  ([0014](0014-gepa-targets-repro-pipeline.md)), so `ollama` /
  `qwen3.5:9b` runs the GEPA arm on exactly the stack 003 ran. The
  `reflection_lm` stays a separate, stronger LiteLLM-routed model
  ([0003](0003-model-routing.md)); nothing here changes that.
- **`AMemGEPAAdapter` installs `_patch_ollama_chat()` when
  `backend == "ollama"`.** Idempotent, same function `just reproduce`
  uses, so baseline and GEPA arm share one generation setting. A test
  asserts `think=False` reaches `ollama.chat`.
- **Progress lines, same shape as the reproduction's.** The adapter
  reuses `paper_repro._ProgressTracker`: one line per `progress_log_every`
  turns during a build (done/total, last, avg(n=), ETA, elapsed, plus the
  Ollama prefill/eval token breakdown when available), one per N questions
  during QA, a slow-call warning at 3x the running average, and after
  every `evaluate()` a run-level line: builds, cache hits, resumed builds,
  avg build time, questions, LLM calls, `metric_calls=done/budget`,
  elapsed, and a rough ETA from the budget at the run's average pace
  (explicitly labelled rough: cache-hit and acceptance rates drift over a
  run). `run_gepa_optimize.py` now configures logging to the console and
  to `results/<label>/amem_gepa.log` for `tail -f` on a background run.
- **Mid-build checkpoints.** Every `build_checkpoint_every_turns` (default
  20) the in-progress build writes, under its build-cache directory:
  one pickle holding `{turns_done, evo_cnt, memories, LLM-call records}`
  (one file so those can never disagree) and the retriever via the
  vendored `save()` (same format as the final cache, so the resumed build
  continues with the exact retriever state rather than one re-derived
  from notes, which `load_from_local_memory` builds with a different
  document format). Files are written to temp names and `os.replace`d.
  On the next build of that (candidate, conversation): restore, verify
  `len(retriever.corpus) == len(memories)`, resume at `turns_done`; any
  inconsistency discards the checkpoint and rebuilds from scratch (a
  rebuild is faithful; a format-changing fallback is not). A finished
  build deletes its partial. Pre-interruption LLM-call records are kept
  so reflection over a resumed build still sees the whole trace.
- **Resumable accounting.** The adapter persists its counters to
  `results/<label>/adapter_stats.json` after every evaluation and reloads
  them when the file exists, so `summary.json` reports whole-run totals
  (CLAUDE.md's "Cost" section) across interruptions. The run plan prints
  whether it is resuming, the checkpoint cadence, and the exact
  `--run-label` to rerun with; `touch results/<label>/gepa.stop` is
  GEPA's own graceful stop.

## Consequences

- Resume granularity for GEPA is now: iteration (GEPA state) → build
  (cache) → `build_checkpoint_every_turns` turns (partial). An interrupted
  QA pass over a batch still re-answers that batch's questions after
  resume (builds are cached, questions are ~3 orders of magnitude cheaper
  per 0015) -- not worth a fourth layer yet.
- Checkpoint cost: a full re-pickle of the conversation's notes plus the
  retriever matrix every 20 turns -- a few MB each at LoCoMo scale, as in
  0007. Disposable, under gitignored `results/`.
- The run-level ETA is deliberately coarse. Use the per-build ETA (which
  is well-calibrated after a handful of turns) to judge whether a run is
  healthy; use the run-level one only for "hours or days".
- Still not measured: a real `qwen3.5:9b` build time through this
  adapter. The first real `just gepa-smoke` / `just gepa-optimize` run
  should record avg build time and the checkpoint overhead in its
  experiment doc, and 0015's "truncated train instances" deferral should
  be revisited against that number.
- Scope: `backend == "ollama"` for the thinking fix (an Ollama-only API
  concept); everything else is backend-agnostic.
