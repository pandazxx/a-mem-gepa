"""GEPAAdapter for optimizing A-MEM's two prompts jointly (docs/decisions/0002).

Interface confirmed against gepa-ai/gepa's gepa/core/adapter.py (a Protocol,
not a base class requiring inheritance):

    Candidate = dict[str, str]  # component name -> prompt text

    def evaluate(self, batch: list[DataInst], candidate: Candidate,
                 capture_traces: bool = False) -> EvaluationBatch[Trajectory, RolloutOutput]: ...

    def make_reflective_dataset(self, candidate: Candidate,
                                 eval_batch: EvaluationBatch, components_to_update: list[str]
                                 ) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...

Candidate keys used here: "note_construction" and "evolution", matching the
two prompts PromptInjectableMemorySystem (amem_adapter.py) accepts.

Not implemented yet -- depends on the LoCoMo dataset loader and metrics
(src/amem_gepa/datasets/locomo.py, src/amem_gepa/metrics.py), which are
milestone-2 work (see the milestone plan in the project discussion, not yet
committed to a doc). Signatures are final; bodies are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch

from amem_gepa.amem_adapter import PromptInjectableMemorySystem


@dataclass
class LoCoMoInstance:
    """One (conversation, question) pair. `conversation_id` must map to an
    entry in the split manifest (configs/locomo_split.json, docs/decisions/0004)."""

    conversation_id: str
    turns: list[dict]
    question: str
    gold_answer: str
    category: str


@dataclass
class LoCoMoTrajectory:
    predicted_answer: str
    note_construction_calls: list[dict]
    evolution_calls: list[dict]
    error: str | None = None


class AMemGEPAAdapter:
    def __init__(self, llm_model: str, llm_api_base: str | None = None):
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base

    def _build_memory_system(self, candidate: dict[str, str]) -> PromptInjectableMemorySystem:
        return PromptInjectableMemorySystem(
            note_construction_prompt=candidate["note_construction"],
            evolution_prompt=candidate["evolution"],
            llm_model=self.llm_model,
            llm_api_base=self.llm_api_base,
        )

    def evaluate(
        self,
        batch: list[LoCoMoInstance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[LoCoMoTrajectory, str]:
        # TODO(milestone-2): for each instance, replay its conversation turns
        # through a fresh PromptInjectableMemorySystem.add_note(), answer the
        # question via retrieval + QA prompt, score with metrics.py, and
        # (if capture_traces) keep enough of the trace for
        # make_reflective_dataset to build a useful reflection example.
        raise NotImplementedError

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[LoCoMoTrajectory, str],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        # TODO(milestone-3): turn eval_batch.trajectories into per-component
        # reflective examples -- e.g. for "note_construction", show cases
        # where a wrong/refused answer traces back to a poor keywords/context/
        # tags extraction; for "evolution", cases where the wrong memory got
        # linked/strengthened.
        raise NotImplementedError
