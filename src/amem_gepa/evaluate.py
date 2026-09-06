"""Shared eval loop, used by both scripts/run_baseline.py (fixed A-MEM
prompts) and scripts/run_eval.py (any candidate) -- and, later, by
gepa_adapter.py's evaluate(), since a GEPA rollout is the same replay-then-
answer-then-score loop over a different prompt candidate.

One conversation -> one fresh PromptInjectableMemorySystem (never shared
across conversations, so there's no cross-conversation memory leakage) ->
replay every turn via add_note() -> for each of that conversation's
questions, retrieve top-k memories via search_agentic() (no extra LLM call,
docs/decisions/0006) -> answer via the fixed qa_answer.txt prompt (one LLM
call) -> score with metrics.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol

from amem_gepa.datasets.locomo import LoCoMoInstance, LoCoMoTurn
from amem_gepa.metrics import score_batch

if TYPE_CHECKING:
    from amem_gepa.amem_adapter import PromptInjectableMemorySystem


class MemorySystemFactory(Protocol):
    def __call__(self, note_construction_prompt: str, evolution_prompt: str) -> "PromptInjectableMemorySystem": ...


@dataclass
class InstanceResult:
    conversation_id: str
    question: str
    category: int
    prediction: str
    gold_answer: Optional[str]
    adversarial_answer: Optional[str]
    score: float


@dataclass
class EvalResult:
    instance_results: list[InstanceResult] = field(default_factory=list)
    summaries: dict = field(default_factory=dict)


def _default_memory_system_factory(
    llm_model: str, llm_api_base: Optional[str], embedding_model: str
) -> MemorySystemFactory:
    # Imported lazily: pulls in agentic_memory (ChromaDB, sentence-transformers,
    # ...) which tests that inject a fake factory shouldn't need installed.
    from amem_gepa.amem_adapter import PromptInjectableMemorySystem

    def factory(note_construction_prompt: str, evolution_prompt: str) -> "PromptInjectableMemorySystem":
        return PromptInjectableMemorySystem(
            note_construction_prompt=note_construction_prompt,
            evolution_prompt=evolution_prompt,
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            model_name=embedding_model,
        )

    return factory


def replay_conversation(memory_system: PromptInjectableMemorySystem, turns: list[LoCoMoTurn]) -> None:
    for turn in turns:
        memory_system.add_note(f"{turn.speaker}: {turn.text}", time=turn.date_time)


def _format_retrieved_memories(memories: list[dict]) -> str:
    if not memories:
        return "(no memories retrieved)"
    lines = []
    for mem in memories:
        ts = mem.get("timestamp", "")
        context = mem.get("context", "")
        lines.append(f"- [{ts}] {mem['content']}" + (f" (context: {context})" if context else ""))
    return "\n".join(lines)


def answer_question(
    memory_system: PromptInjectableMemorySystem,
    qa_prompt_template: str,
    question: str,
    k: int = 10,
) -> str:
    retrieved = memory_system.search_agentic(question, k=k)
    prompt = qa_prompt_template.format(
        retrieved_memories=_format_retrieved_memories(retrieved), question=question
    )
    return memory_system.llm_controller.llm.get_completion(prompt, response_format=None)


def evaluate_candidate(
    instances: list[LoCoMoInstance],
    note_construction_prompt: str,
    evolution_prompt: str,
    qa_prompt_template: str,
    llm_model: str,
    llm_api_base: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    k: int = 10,
    memory_system_factory: Optional[MemorySystemFactory] = None,
    n_bootstrap_resamples: int = 10_000,
) -> EvalResult:
    """Evaluate one (note_construction_prompt, evolution_prompt) candidate
    against `instances`. `memory_system_factory` is injectable so tests can
    substitute a fake memory system instead of a real A-MEM + LLM stack."""
    factory = memory_system_factory or _default_memory_system_factory(
        llm_model, llm_api_base, embedding_model
    )

    by_conversation: dict[str, list[LoCoMoInstance]] = defaultdict(list)
    for inst in instances:
        by_conversation[inst.conversation_id].append(inst)

    instance_results: list[InstanceResult] = []
    for conv_id, conv_instances in by_conversation.items():
        memory_system = factory(note_construction_prompt, evolution_prompt)
        replay_conversation(memory_system, conv_instances[0].turns)

        for inst in conv_instances:
            prediction = answer_question(memory_system, qa_prompt_template, inst.question, k=k)
            score = score_batch(
                [prediction],
                [inst.category],
                [inst.gold_answer],
                [inst.adversarial_answer],
                n_resamples=1,
            )["aggregate"].mean
            instance_results.append(
                InstanceResult(
                    conversation_id=conv_id,
                    question=inst.question,
                    category=inst.category,
                    prediction=prediction,
                    gold_answer=inst.gold_answer,
                    adversarial_answer=inst.adversarial_answer,
                    score=score,
                )
            )

    summaries = score_batch(
        [r.prediction for r in instance_results],
        [r.category for r in instance_results],
        [r.gold_answer for r in instance_results],
        [r.adversarial_answer for r in instance_results],
        n_resamples=n_bootstrap_resamples,
    )
    return EvalResult(instance_results=instance_results, summaries=summaries)
