"""Sequential single-prompt GEPA training (docs/decisions/0019).

The first full GEPA run (docs/experiments/004) let GEPA's default round-robin
selector rewrite a different one of the four memory prompts each iteration,
so the accepted candidates mixed changes to all four and the +5.6 validation
gain did not survive on the test split. This module supports the alternative:
optimize ONE component per run while the other three stay byte-identical to
that run's seed candidate -- coordinate ascent over the four prompts, one
stage per prompt, the next stage seeded from the previous stage's winner.

Three pieces, all small:

- FixedComponentSelector: a gepa ReflectionComponentSelector that always
  names the same component. gepa builds a child as `parent.copy()` with only
  the selected components' texts replaced (gepa/proposer/reflective_mutation/
  reflective_mutation.py), and merge proposals are off in our runs, so this
  alone guarantees the frozen components never change.
- frozen_violations(): the belt-and-braces check the adapter applies to every
  candidate it is asked to evaluate -- a candidate that differs from the seed
  on a frozen component scores 0 without spending a build, exactly like a
  template that fails placeholder validation (repro_injection).
- load_seed_candidate(): a seed directory plus per-component file overrides,
  so stage N+1 can start from stage N's committed
  `src/amem_gepa/prompts/<component>.v<N>.txt` (CLAUDE.md's prompt-versioning
  rule) rather than from a gitignored results/ directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from amem_gepa.repro_injection import COMPONENTS, load_baseline_candidate

# The order the stages run in. note_construction first: it is the upstream
# step (every evolution prompt consumes its keywords/context, and its output
# is what retrieval embeds), it fires on every turn, and its reflection
# dossier is the most directly attributable one (an evidence turn that was
# never retrieved points at the note written for it). The three evolution
# prompts follow in pipeline order. docs/decisions/0019 has the reasoning.
COMPONENT_ORDER: tuple[str, ...] = (
    "note_construction",
    "evolution_decision",
    "evolution_strengthen",
    "evolution_update_neighbors",
)
assert set(COMPONENT_ORDER) == set(COMPONENTS)


def _check_component(component: str) -> None:
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}; expected one of {list(COMPONENTS)}")


class FixedComponentSelector:
    """gepa `ReflectionComponentSelector` (a Protocol, duck-typed) that
    ignores the round-robin bookkeeping in GEPAState and always selects the
    one component this run is optimizing."""

    def __init__(self, component: str):
        _check_component(component)
        self.component = component

    def __call__(self, state, trajectories, subsample_scores, candidate_idx, candidate) -> list[str]:
        return [self.component]

    def __repr__(self) -> str:
        return f"FixedComponentSelector({self.component!r})"


def frozen_components(seed: Mapping[str, str], component: str) -> dict[str, str]:
    """The seed's texts for every component except `component` -- what the
    adapter holds every evaluated candidate to."""
    _check_component(component)
    return {name: seed[name] for name in COMPONENTS if name != component}


def frozen_violations(candidate: Mapping[str, str], frozen: Mapping[str, str]) -> list[str]:
    """Human/LLM-readable problems, empty if every frozen component is
    byte-identical to its frozen text. Same shape as
    repro_injection.validate_candidate's output so the adapter can treat the
    two identically."""
    free = sorted(set(COMPONENTS) - set(frozen))
    problems = []
    for name in COMPONENTS:
        if name in frozen and candidate.get(name) != frozen[name]:
            problems.append(
                f"{name}: this component is FROZEN for this run and must stay byte-identical to the "
                f"seed candidate; only {', '.join(free) or 'nothing'} may change (docs/decisions/0019)"
            )
    return problems


def changed_components(candidate: Mapping[str, str], seed: Mapping[str, str]) -> list[str]:
    return [name for name in COMPONENTS if candidate.get(name) != seed.get(name)]


def parse_seed_override(spec: str) -> tuple[str, Path]:
    """`component=path/to/prompt.txt` -> (component, Path). Raises ValueError
    on a malformed spec or an unknown component."""
    if "=" not in spec:
        raise ValueError(f"seed override must look like component=path, got {spec!r}")
    component, path = spec.split("=", 1)
    component = component.strip()
    _check_component(component)
    if not path.strip():
        raise ValueError(f"seed override for {component} has an empty path")
    return component, Path(path.strip())


def load_seed_candidate(seed_dir: Path, overrides: Mapping[str, Path] | None = None) -> dict[str, str]:
    """The four prompt files under `seed_dir` (baseline_repro/ by default,
    or a previous stage's best_candidate/), with individual components
    replaced by the files in `overrides` -- how a stage picks up the
    previous stage's committed winner (prompts/<component>.v<N>.txt)."""
    candidate = load_baseline_candidate(Path(seed_dir))
    for component, path in (overrides or {}).items():
        _check_component(component)
        candidate[component] = Path(path).read_text()
    return candidate
