# 0012: Reject fabricated evolution decisions before they corrupt memory

## Status

Accepted (2026-09-09)

## Context

Reading a real `just demo` trace (turn 11/15, `conv-30`, Llama 3.2:1b) showed
an evolution decision that was syntactically valid JSON but semantically
fabricated:

```
should_evolve=True actions=['strengthen', 'update_neighbor']
suggested_connections=['memory_index_0', 'memory_index_1', 'memory_index_2', 'memory_index_3']
tags_to_update=['keywords_0', 'keywords_1', 'keywords_2', 'keywords_3',
                'memory_tags_0', 'memory_tags_1', 'memory_tags_2', 'memory_tags_3']
```

`suggested_connections` should be real memory IDs (UUIDs); `tags_to_update`
should be real tag words. Instead both look like the model echoed
placeholder/field-name-shaped tokens rather than reasoning about the actual
neighbor notes it was shown. Upstream's `process_memory`
(`external/a-mem/agentic_memory/memory_system.py`) has no guard against
this -- for a `"strengthen"` action it applies the decision verbatim:

```python
note.links.extend(suggest_connections)   # 4 links to memory IDs that don't exist
note.tags = new_tags                     # note's real tags overwritten with garbage
```

Since a note's `context`/`tags` are what actually get shown to the QA-
answering step later (`_format_retrieved_memories`, `evaluate.py`), this is
a plausible, concrete contributor to the poor single_hop/temporal/
multi_hop/open_domain scores from milestone 2 -- not (or not only) "the
model phrases answers badly," but "the memory it's answering from was
quietly corrupted several turns earlier."

## Decision

- `amem_adapter.py`'s evolution-call wrapper (renamed
  `_EvolutionGuardBackend`, was `_EvolutionTraceBackend`,
  docs/decisions/0011) now validates before returning the response to
  upstream: if `should_evolve` is true and `suggested_connections` is
  non-empty but none of its entries match a real key in `self.memories`,
  the decision is rejected -- `should_evolve` forced to `False`, `actions`
  cleared -- and the corrected JSON is what upstream actually parses.
  A one-line `[evolution guard] rejected: ...` message always prints when
  this fires.
- This is **always on**, not gated by `trace` -- it's a correctness fix,
  not a debugging aid. `just baseline` gets the same protection as
  `just demo`, just without the rest of 0011's verbose per-call dump.
- Deliberately narrow: only checks whether `suggested_connections`
  references anything real. `tags_to_update`'s *content* can't be
  independently verified (any string could be a legitimate tag), so the
  fix relies on rejecting the whole decision when the connections signal
  looks fabricated, rather than trying to salvage individual fields.
  `update_neighbor`'s `new_context_neighborhood`/`new_tags_neighborhood`
  aren't validated at all yet -- they're positional against an internal
  neighbor list this wrapper doesn't have access to (computed inside
  upstream's `process_memory`, not exposed).

## Consequences

- A model that reliably produces fabricated `suggested_connections` will
  now never get to `"strengthen"` a note -- effectively disabling that
  half of evolution for it, rather than letting it corrupt data. This
  trades "evolution does something" for "evolution does nothing incorrect
  data-corrupting" when the model can't be trusted -- the right tradeoff
  given note quality feeds directly into QA answers.
- `update_neighbor`'s neighbor-content corruption risk (plausible, not yet
  confirmed) is explicitly unguarded -- a real gap, not an oversight to be
  silently left unmentioned. Closing it would need either re-deriving the
  neighbor list from the prompt text or exposing it from
  `find_related_memories` some other way.
- Once GEPA (milestone 3+) starts mutating the evolution prompt, this
  guard's rejection rate becomes a useful signal on its own: a candidate
  prompt that gets rejected constantly is one whose evolution instructions
  the model can't follow, independent of whatever the eventual QA-score
  fitness says.
