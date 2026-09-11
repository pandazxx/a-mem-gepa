# 0008: A small demo run before the full reproduction

## Status

Accepted (2026-09-07)

## Context

The milestone-2 reproduction sanity check (docs/decisions/0005) took
multiple days against Llama 3.2:1b and still didn't finish before being
interrupted -- even with resumable checkpointing (docs/decisions/0007), a
full run (3 test conversations, ~600 questions, ~2 LLM calls per turn
across hundreds of turns each) is a bad way to first find out whether the
adapter, prompts, and eval loop actually work end to end. A pipeline bug
discovered on day 3 of a resumed run is much more expensive to find than
one discovered in ten minutes.

## Decision

- Add `scripts/run_demo.py` (`just demo`): one conversation, truncated to
  its first `--max-turns` turns (default 15), and up to `--max-questions`
  QA pairs (default 5) whose evidence falls entirely within that truncated
  window -- so the demo is actually answerable from what got replayed, not
  silently testing against missing context.
- Questions are chosen round-robin across categories (`load_demo_sample`,
  `datasets/locomo.py`) so a 5-question demo has a shot at showing one of
  each, which is more useful for a human eyeballing predicted-vs-expected
  output than 5 open-domain questions in a row.
- Defaults to the **first train-split conversation**, not val/test --
  looking at demo output during development shouldn't mean looking at data
  the real reproduction/GEPA runs are held out against.
- `run_demo.py` prints every question/prediction/expected-answer/score, not
  just an aggregate, since the point is a human judgment call ("does this
  look like it's working"), not a statistic.
- Workflow going forward: `just demo` first, always. Only run the full
  `just baseline` (docs/decisions/0005) once the demo output looks sane.

## Consequences

- The demo is explicitly not a reproduction and produces no number worth
  putting in `docs/experiments/` -- it's a development/debugging tool, nothing
  in it should be cited as a result.
- `evaluate_candidate`'s existing `run_label`/checkpointing (0007) applies
  to the demo too for free (`run_label="demo"`), so even a demo run is
  resumable if interrupted -- though at 15 turns / 5 questions it shouldn't
  need to be.
- Truncating a conversation to its first N turns means the demo only
  exercises early-conversation memory evolution (few existing notes to link
  against) -- it's a sanity check on plumbing, not a check that evolution
  behaves sensibly once a conversation has hundreds of notes. That gap is
  intentional; closing it is what the full reproduction is for.
