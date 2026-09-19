"""Runtime prompt injection into the paper-reproduction pipeline
(docs/decisions/0014).

The four GEPA-optimizable prompts live as module-level constants in
external/agentic-memory-repro/llm_text_parsers.py and are imported *by name*
into memory_layer_robust's namespace, where every call site resolves them
(RobustMemoryNote.analyze_content, RobustAgenticMemorySystem.process_memory).
Injecting a candidate therefore means patching those names on the
memory_layer_robust module object -- no edit to the read-only vendored
submodule, same category of wrapper as paper_repro.py's file-path
redirection (docs/decisions/0013).

Also here:
- candidate validation: the templates are consumed via str.format(), so an
  evolved prompt that drops a required {placeholder} or invents a new one
  raises KeyError/IndexError deep inside the vendored code, mid-build,
  after real API spend. validate_candidate() catches that up front so the
  adapter can score the candidate 0 with actionable feedback instead.
- RecordingLLM: a wrapper around the repro pipeline's LLM controller that
  logs every (prompt, response) pair and classifies which template produced
  it, so make_reflective_dataset can attribute failures to components.

NOT injected, deliberately (docs/decisions/0014): FOCUSED_KEYWORDS_PROMPT
(an error-recovery fallback, not a memory-management decision), the
generate_query_llm retrieval prompt, and the category-specific QA prompts
(docs/decisions/0006's frozen-QA reasoning applies to all of them).
"""

from __future__ import annotations

import hashlib
import json
import string
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from amem_gepa.paper_repro import ensure_repro_repo_importable

BASELINE_REPRO_DIR = Path(__file__).resolve().parent / "prompts" / "baseline_repro"

# component name -> the constant it replaces in memory_layer_robust's namespace
COMPONENT_TO_CONSTANT = {
    "note_construction": "ANALYZE_CONTENT_PROMPT",
    "evolution_decision": "EVOLUTION_DECISION_PROMPT",
    "evolution_strengthen": "STRENGTHEN_DETAILS_PROMPT",
    "evolution_update_neighbors": "UPDATE_NEIGHBORS_PROMPT",
}

COMPONENTS = tuple(COMPONENT_TO_CONSTANT)

# Exactly the str.format() fields each call site supplies (read from
# memory_layer_robust.py's analyze_content/process_memory). A candidate must
# use all of them (dropping one starves the LLM of that input) and may not
# add new ones (str.format raises KeyError on an unknown field).
REQUIRED_PLACEHOLDERS: dict[str, frozenset[str]] = {
    "note_construction": frozenset({"content"}),
    "evolution_decision": frozenset({"context", "content", "keywords", "nearest_neighbors_memories"}),
    "evolution_strengthen": frozenset({"content", "keywords", "nearest_neighbors_memories"}),
    "evolution_update_neighbors": frozenset(
        {"content", "context", "nearest_neighbors_memories", "max_neighbor_idx", "neighbor_count"}
    ),
}


def load_baseline_candidate(baseline_dir: Path = BASELINE_REPRO_DIR) -> dict[str, str]:
    """The paper's verbatim prompts as a GEPA seed candidate. The files are
    generated from llm_text_parsers' constants (not hand-copied) and
    tests/test_repro_injection.py asserts byte-equality against the vendored
    submodule, so a silent drift after a submodule bump fails loudly."""
    return {name: (baseline_dir / f"{name}.txt").read_text() for name in COMPONENTS}


def candidate_hash(candidate: dict[str, str]) -> str:
    """Stable content hash identifying a candidate, used to key the
    per-(candidate, conversation) memory-build cache (docs/decisions/0015)."""
    canonical = json.dumps({k: candidate[k] for k in sorted(candidate)}, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _template_fields(template: str) -> tuple[set[str], list[str]]:
    """(named fields, errors) found by parsing `template` as a str.format
    template -- the same parser .format() itself uses, so anything that
    breaks here breaks identically at the real call site."""
    fields: set[str] = set()
    errors: list[str] = []
    try:
        for _, field_name, _, _ in string.Formatter().parse(template):
            if field_name is None:
                continue
            if field_name == "" or field_name.isdigit():
                errors.append(
                    "uses a positional placeholder ('{}' or '{0}') -- the call site "
                    "formats with keyword arguments only"
                )
                continue
            # "{content[0]}"/"{content.x}" index into the argument; keep the root name.
            fields.add(field_name.split(".")[0].split("[")[0])
    except ValueError as e:
        errors.append(f"is not a valid str.format template: {e} (literal braces must be doubled: '{{{{' '}}}}')")
    return fields, errors


def validate_candidate(candidate: dict[str, str]) -> list[str]:
    """Human/LLM-readable problems that would make this candidate crash or
    misbehave at .format() time. Empty list = safe to inject."""
    problems: list[str] = []
    for component in COMPONENTS:
        if component not in candidate or not str(candidate[component]).strip():
            problems.append(f"{component}: missing or empty prompt text")
            continue
        required = REQUIRED_PLACEHOLDERS[component]
        fields, errors = _template_fields(candidate[component])
        for err in errors:
            problems.append(f"{component}: {err}")
        if errors:
            continue
        def braced(names: frozenset[str] | set[str]) -> list[str]:
            return sorted("{" + name + "}" for name in names)

        missing = required - fields
        extra = fields - required
        if missing:
            problems.append(
                f"{component}: missing required placeholder(s) {braced(missing)} "
                f"-- the template must contain exactly these placeholders: {braced(required)}"
            )
        if extra:
            problems.append(
                f"{component}: unknown placeholder(s) {braced(extra)} "
                f"-- the call site only supplies {braced(required)}; "
                "anything else raises KeyError at format time"
            )
    return problems


@contextmanager
def injected_prompts(candidate: dict[str, str]):
    """Swap the candidate's four prompts into memory_layer_robust's
    namespace for the duration of the block, restoring the originals after
    -- including on exception, so a failed build can't leak an evolved
    prompt into a later baseline evaluation in the same process."""
    ensure_repro_repo_importable()
    import memory_layer_robust as mlr

    originals = {name: getattr(mlr, name) for name in COMPONENT_TO_CONSTANT.values()}
    try:
        for component, constant in COMPONENT_TO_CONSTANT.items():
            setattr(mlr, constant, candidate[component])
        yield
    finally:
        for name, value in originals.items():
            setattr(mlr, name, value)


# ---------------------------------------------------------------------------
# LLM-call recording (for GEPA reflection traces)
# ---------------------------------------------------------------------------

# Frozen prompts we still want to label in traces, keyed by a distinctive
# static prefix (test_advanced_robust.py's inline f-strings and
# llm_text_parsers.FOCUSED_KEYWORDS_PROMPT).
_FROZEN_PREFIXES = {
    "query_keywords": "Given the following question, generate several keywords",
    "qa_answer": "Based on the context:",
    "relevant_parts": "Given the following conversation memories and a question",
    "keywords_fallback": "List exactly 5 keywords",
}

_PROMPT_CHARS_KEPT = 6000  # bound trace size; neighbor blocks can be large


def _static_prefix(template: str) -> str:
    """Template text before its first placeholder -- what a formatted prompt
    is guaranteed to start with."""
    return template.split("{", 1)[0]


@dataclass
class LLMCallRecord:
    kind: str
    prompt: str
    response: str
    temperature: float

    def to_dict(self) -> dict:
        return {"kind": self.kind, "prompt": self.prompt, "response": self.response, "temperature": self.temperature}


class RecordingLLM:
    """Wraps the repro pipeline's inner controller (the object exposing
    get_completion) to log every call, classified by which template's static
    prefix the prompt starts with. Longest-prefix wins, so two candidate
    templates sharing an opening line still classify correctly as long as
    they diverge before their first placeholder."""

    def __init__(self, inner, candidate: dict[str, str]):
        self._inner = inner
        self.records: list[LLMCallRecord] = []
        prefixes = {component: _static_prefix(text) for component, text in candidate.items()}
        prefixes.update(_FROZEN_PREFIXES)
        # Empty prefixes (template starts with a placeholder) can't be matched.
        self._prefixes = sorted(
            ((kind, prefix) for kind, prefix in prefixes.items() if prefix),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )

    def classify(self, prompt: str) -> str:
        for kind, prefix in self._prefixes:
            if prompt.startswith(prefix):
                return kind
        return "other"

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        response = self._inner.get_completion(prompt, temperature=temperature)
        self.records.append(
            LLMCallRecord(
                kind=self.classify(prompt),
                prompt=prompt[:_PROMPT_CHARS_KEPT],
                response=response,
                temperature=temperature,
            )
        )
        return response

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass
class BuildTrace:
    """Everything recorded while replaying one conversation through one
    candidate's prompts. Persisted next to the memory-build cache so a
    cache hit still has construction/evolution traces available for
    reflection (a rebuilt trace would cost a full memory build)."""

    conversation_id: str
    candidate_hash: str
    records: list[LLMCallRecord] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "conversation_id": self.conversation_id,
                "candidate_hash": self.candidate_hash,
                "records": [r.to_dict() for r in self.records],
            }
        )

    @classmethod
    def from_json(cls, text: str) -> BuildTrace:
        data = json.loads(text)
        return cls(
            conversation_id=data["conversation_id"],
            candidate_hash=data["candidate_hash"],
            records=[LLMCallRecord(**r) for r in data["records"]],
        )
