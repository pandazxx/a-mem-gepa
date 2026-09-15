# 0007: Resumable, progress-reporting eval runs

## Status

Accepted (2026-09-06)

## Context

A real `just baseline` run against Llama 3.2:1b (docs/decisions/0005) took
12+ hours and still hadn't finished, with no visible progress in the
meantime. Each conversation replay is ~2 LLM calls per turn (note
construction + evolution, docs/decisions/0006) and conversations run
100-400+ turns -- on a slow local model, losing that to a Ctrl-C, crash, or
laptop sleep is expensive, and this will only get more expensive once GEPA
(milestone 3+) is doing many rollouts of the same shape.

## Decision

- **Progress reporting** (`progress.py`): one line per turn replayed and per
  question answered, with elapsed time, rate, and ETA. No external
  dependency (no `tqdm`) -- plain prints are enough and stay readable when
  piped to a log file.
- **Checkpoint granularity is per-turn, not per-conversation**
  (`checkpoint.py`, `evaluate.py`). Rejected coarser per-conversation
  checkpointing (simpler, but a single conversation can itself run for
  hours, so it wouldn't have prevented the exact problem being fixed here)
  in favor of: after every `add_note()` call, snapshot the memory system's
  full note state (`amem_adapter.snapshot_notes()`) and how many turns are
  done to `results/<run_label>/checkpoints/<conv_id>.json`. On resume,
  `restore_notes()` repopulates the memory system and ChromaDB retriever
  directly from that snapshot -- no LLM calls -- and replay continues from
  the next un-processed turn.
- **Answered questions are checkpointed too**, independently, as an
  append-only JSONL log (`results/<run_label>/<split>_predictions.jsonl`).
  Already-answered questions are skipped on resume rather than re-asked.
- Both are opt-in via an `evaluate_candidate(..., run_label=...)` parameter.
  `run_baseline.py`/`run_eval.py` always pass one (so `just baseline`/
  `just eval` are resumable by default -- rerunning the same command after
  an interruption continues instead of restarting); the unit tests that
  inject a fake `memory_system_factory` don't, and run exactly as before.

## Consequences

- `results/<run_label>/` (gitignored, per CLAUDE.md) is now load-bearing
  for resuming a run, not just a place final output happens to land --
  deleting it mid-run means starting that run over.
- Checkpoint files grow with each conversation's note count (a full
  re-snapshot on every turn, not a diff) -- fine at LoCoMo's scale (single
  conversations, hundreds of notes), would need revisiting for much larger
  conversations.
- Snapshot/restore only covers what `add_note()`/`process_memory()`
  produce (`amem_adapter._NOTE_FIELDS`) -- if a future A-MEM submodule bump
  adds new `MemoryNote` fields, `_NOTE_FIELDS` needs a manual update or a
  resumed run will silently drop them.
