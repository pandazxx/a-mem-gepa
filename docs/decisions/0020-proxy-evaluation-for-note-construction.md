# 0020: Train note construction without full rebuilds — re-note only the question-relevant notes on a seed-built memory

## Status

**Proposed, implemented behind a flag** (2026-11-26; proposed 2026-11-22).
Available as `--proxy evidence|competitors` on `just gepa-optimize-component
note_construction` and `just gepa-test-eval ... --split val`
(`AMemGEPAAdapter(proxy_mode=...)`). **Not yet accepted as a training
default**: that needs the calibration experiment below, which is now
cheaper than written -- stage 1's winner ([experiments/005](../experiments/005-gepa-stage1-note-construction-qwen3.5-9b.md),
`prompts/note_construction.v1.txt`) differs from the paper's prompts in
note construction only and already has a full-build validation score
(0.3798 vs seed 0.3221), so the calibration is two proxy runs on
validation against numbers already on disk:

```
just gepa-test-eval <stage-1 best_candidate> --split val --proxy evidence --yes
just gepa-test-eval <stage-1 best_candidate> --split val --proxy competitors --yes
```

(~10 min and ~2 h respectively, no new builds.) Accept if both tiers rank
the winner above the seed and the `competitors` tier's delta is within the
paired-bootstrap interval of the full-build delta; record the gap either
way in the experiment doc. Depends on
[0019](0019-sequential-single-prompt-training.md)'s single-component
stages: this applies to the `note_construction` stage only.

## Context

A GEPA rollout rebuilds a conversation's memory from scratch for every new
candidate because the candidate's prompts are what build it
([0015](0015-gepa-training-method.md)). In
[experiments/004](../experiments/004-gepa-v1-qwen3.5-9b.md) that was 96%
of 6.3 days: ~89 min and ~1,900 LLM calls per build, against ~6 s per
question. A single-component stage at 800 metric calls (0019) is ~55
iterations of 2 builds plus ~4 full-validation passes of 2 builds each --
on the order of a week per stage.

The user's question: when only the note-construction prompt is being
trained, can we build each conversation **once with the seed prompt**, and
for a given question re-construct only the notes that matter to it with
the candidate prompt?

### What note construction actually changes

Per turn, the vendored pipeline (`memory_layer_robust.py`) does:

1. `RobustMemoryNote(content, llm_controller)` → `analyze_content()` runs
   `ANALYZE_CONTENT_PROMPT` **once** and sets the note's `keywords`,
   `context`, `tags`. The raw turn text is stored verbatim regardless.
2. `process_memory(note)` → finds 5 neighbors by embedding `note.content`
   (the raw text -- *not* the keywords), then runs the three evolution
   prompts, whose inputs include the new note's `context`/`keywords`.
   Outcomes: links on the new note, rewritten `context`/`tags` on
   neighbors.
3. `retriever.add_documents(["content:" + content + " context:" + context
   + " keywords: " + ... + " tags: " + ...])` → one embedding per note
   whose text is dominated by what step 1 produced.

At answer time: an LLM turns the question into keywords; the retriever
ranks **all** notes' embeddings against them; the top-k plus their linked
neighbors become the QA context.

So the note-construction prompt affects a question's answer through three
channels: (a) the evidence notes' own embeddings, i.e. whether they rank
in the top-k; (b) every *other* note's embedding, i.e. how strong the
competition is; (c) the evolution prompts' inputs, i.e. which links and
neighbor rewrites happened. Channel (a) is direct and per-note. (b) and
(c) are whole-memory effects.

### Why the mechanics make a partial rebuild possible

- `add_note` stores exactly one note per turn into an insertion-ordered
  dict, and the adapter replays `item.turns` in order, so **note index
  i == turn i**, and LoCoMo's per-question evidence `dia_id`s map to note
  indices through `turns[i].dia_id`.
- `analyze_content` is a static method reading the module-level prompt
  constant, so running it under `repro_injection.injected_prompts(candidate)`
  re-notes a single turn with the candidate's prompt: **one LLM call**.
- `SimpleEmbeddingRetriever` keeps a parallel `corpus` list and
  `embeddings` matrix; `search()` uses only the matrix. Replacing row i
  (and `corpus[i]`) with the re-noted document is a local edit. The
  document format is detectable from `corpus[i]` (`"content:"` prefix for
  `add_documents`' format; `" , "` separator after a `consolidate_memories`
  rebuild).
- The seed build of every train/val conversation is already in
  `results/gepa_cache/` from 004 (memories pickle + retriever files).

## Decision (proposed)

Add a `proxy_note_construction` evaluation mode to `AMemGEPAAdapter`, used
for the minibatch and validation rollouts of a `note_construction` stage
only. Per (candidate, conversation, question set):

1. Load the **seed** build from the cache (never rebuild).
2. Select the notes to re-construct: the evidence turns of every question
   in the set (channel a), plus -- tier A+ -- the notes the *seed* memory
   retrieves in its top-k for each question (the direct competitors,
   a sample of channel b).
3. Under `injected_prompts(candidate)`, call `analyze_content` for each
   selected turn's raw text; overwrite that note's `keywords`/`context`/
   `tags`; re-embed its retriever row in the same document format it had.
   Links and all other notes stay as the seed built them (channel c is
   deliberately frozen).
4. Answer the questions through the unmodified `answer_question` and score
   as in [0015](0015-gepa-training-method.md).
5. Reflection works unchanged: the re-noting calls are recorded by
   `RecordingLLM` as `note_construction` records containing the evidence
   turn text, which is exactly what `make_reflective_dataset` looks for.

Cost per minibatch group of ~15 questions (004's measured ~4-6 s/call):

| mode | LLM calls | wall-clock | what it sees |
|---|---|---|---|
| full build (today) | ~1,900 + 30 QA | ~90 min | everything |
| tier A: evidence notes only | ~20 + 30 QA | **~4 min** | channel (a) |
| tier A+: evidence + seed top-10 competitors | ~170 + 30 QA | **~15-20 min** | (a) + sampled (b) |
| tier C: re-note every turn, replay the seed's evolution | ~300 + 30 QA | ~30 min | (a) + (b), stale (c) |

Full validation (138 questions, 2 conversations) is ~10 min in tier A and
~2 h in tier A+ (vs ~3 h of builds today). The recommendation is tier A+
for minibatches (where the acceptance decision is made) and tier A for
the frequent validation passes, with the **winner of the stage re-scored
by a full build on validation** before anything is committed.

### What the proxy cannot see, stated up front

- **Competition from un-re-noted notes.** A candidate that makes
  evidence notes more retrievable *and* makes 300 distractors more
  retrievable looks better under the proxy than it is. Tier A+ samples
  the competitors that matter most under the seed; it cannot see
  distractors that become competitive only under the candidate.
- **Evolution inputs.** With candidate keywords/context the frozen
  evolution prompts would have made different links and rewrites. The
  proxy keeps the seed's. For a stage that freezes the evolution
  *prompts* this is the smallest of the three biases, but it is not zero
  -- 004's strengthen rewrite shows links move scores.
- **Consolidation.** Every 100 evolutions the retriever is rebuilt in a
  different document format; the proxy preserves whichever format each
  row has, but a candidate cannot change when consolidation fires.
- **The proxy is a surrogate, and GEPA will optimize the surrogate.** The
  guardrail is the full-build re-score of the stage winner: if proxy gain
  and full-build gain disagree, the stage result is the full-build number
  and the proxy's calibration is recorded in the experiment doc.

### Calibration before acceptance (no training run needed)

004's cache already holds full builds of the seed and of candidates 8 and
10, the two whose lineage changed `note_construction`. The cheap check:
take **only** candidate 8's / 10's note-construction prompt layered on the
seed, score it on validation under tiers A and A+, and compare with a
single full-build validation of that same two-file candidate (2 builds,
~3 h each). If the proxy ranks {seed, 8-note, 10-note} in the same order
as the full builds and the deltas are within the bootstrap noise, accept
this ADR; if not, record the gap and drop the idea. This is one evening
of local compute against a week per stage saved.

### Implementation notes (as built)

- `AMemGEPAAdapter(seed_candidate=..., proxy_mode=..., proxy_competitors_k=...)`.
  Constructor refuses `proxy_mode` without a seed or with
  `frozen_components` other than exactly the three non-note components.
- A candidate takes the proxy path only if it differs from the seed in
  `note_construction` and nothing else; the seed itself takes the normal
  cache path, so the seed's proxy score *is* its full-build score and the
  two scales share their anchor. Frozen-component violations still score
  0 without a build.
- The proxy agent is constructed for the *candidate* (so `RecordingLLM`
  classifies the re-noting calls under the candidate's template prefix and
  reflection sees them as `note_construction` records) and then loaded
  with the seed's cached memories and retriever. Deserialization gives
  fresh objects, so the cached seed is never mutated. If the seed build is
  not cached it is built once through the normal path. If the loaded note
  count does not equal the turn count the proxy is abandoned for a full
  build.
- Competitors are the seed retriever's top-k for the *question text* (no
  LLM call), a sampling heuristic for channel (b), not a reproduction of
  answer-time keyword retrieval.
- Re-embedding writes the row in whichever document format that row has
  (`"content:"`-prefixed from `add_documents`, or the `" , "` form from
  `consolidate_memories`) and keeps `document_ids` consistent.
- `ItemTrajectory.proxy_renoted`, adapter counters `proxy_evaluations` /
  `proxy_renotes` (persisted with the run stats), `stage.json` records the
  proxy mode so a resume under a different mode is refused, and
  `gepa-test-eval` refuses `--proxy` on the test split.

## Consequences

- **Applies to the note-construction stage only.** The three evolution
  prompts act through links and neighbor rewrites, which are relational
  and sequential; there is no per-note surgery that reproduces them. A
  surrogate for those stages would be a different design (e.g. replay the
  seed build up to the evidence turn, then run the candidate's evolution
  on a short window), not this ADR.
- The held-out test comparison ([0018](0018-held-out-test-comparison.md))
  is unchanged: full builds, both arms, every question. The proxy never
  touches a reported number.
- If calibration passes, a `note_construction` stage drops from ~a week
  to ~a day on the local 9B model, which also makes the validation
  fixes 0019's review asked for (4-group minibatches, stricter
  acceptance) affordable.
- New adapter state to test: the turn→note index mapping (assert note
  count == turn count on load, else fall back to a full build), the
  document-format detection per retriever row, and that re-noting never
  mutates the cached seed build on disk (work on a deep copy; the cache
  entry must remain the seed's).
