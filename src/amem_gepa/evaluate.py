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

Progress reporting + resumable checkpointing (progress.py, checkpoint.py):
a full run against a slow local model can take many hours (observed: 12+
hours, still incomplete, on Llama 3.2:1b/M3 Pro), so both are opt-in via
`run_label` -- pass one (as run_baseline.py/run_eval.py do) to get a
resumable, progress-reporting run; omit it (as the unit tests do, with a
fake memory_system_factory) for a plain in-memory run with neither.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol

from amem_gepa.checkpoint import CheckpointStore, ConversationCheckpoint, PredictionStore
from amem_gepa.datasets.locomo import LoCoMoInstance, LoCoMoTurn
from amem_gepa.metrics import score_batch
from amem_gepa.progress import ProgressReporter

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


def replay_conversation(
    memory_system: "PromptInjectableMemorySystem",
    turns: list[LoCoMoTurn],
    checkpoint_store: Optional[CheckpointStore] = None,
    conversation_id: Optional[str] = None,
    progress: Optional[ProgressReporter] = None,
) -> None:
    start_index = 0
    if checkpoint_store is not None:
        checkpoint = checkpoint_store.load(conversation_id)
        if checkpoint is not None:
            memory_system.restore_notes(checkpoint.notes)
            start_index = checkpoint.turns_processed

    for i in range(start_index, len(turns)):
        turn = turns[i]
        memory_system.add_note(f"{turn.speaker}: {turn.text}", time=turn.date_time)
        if progress is not None:
            progress.tick(f"conv={conversation_id} turn={i + 1}/{len(turns)}")
        if checkpoint_store is not None:
            checkpoint_store.save(
                ConversationCheckpoint(
                    conversation_id=conversation_id,
                    turns_processed=i + 1,
                    notes=memory_system.snapshot_notes(),
                    completed=(i + 1 == len(turns)),
                )
            )


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
    memory_system: "PromptInjectableMemorySystem",
    qa_prompt_template: str,
    question: str,
    k: int = 10,
) -> str:
    retrieved = memory_system.search_agentic(question, k=k)
    prompt = qa_prompt_template.format(
        retrieved_memories=_format_retrieved_memories(retrieved), question=question
    )
    return memory_system.llm_controller.llm.get_completion(prompt, response_format=None)


def _score_one(prediction: str, inst: LoCoMoInstance) -> float:
    return score_batch(
        [prediction], [inst.category], [inst.gold_answer], [inst.adversarial_answer], n_resamples=1
    )["aggregate"].mean


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
    run_label: Optional[str] = None,
    split: str = "default",
    results_dir: Path = Path("results"),
) -> EvalResult:
    """Evaluate one (note_construction_prompt, evolution_prompt) candidate
    against `instances`. `memory_system_factory` is injectable so tests can
    substitute a fake memory system instead of a real A-MEM + LLM stack.

    Pass `run_label` to make this resumable: replayed turns and answered
    questions are checkpointed to results/<run_label>/ as they complete, and
    a rerun with the same run_label+split picks up where it left off instead
    of starting over (checkpoint.py)."""
    factory = memory_system_factory or _default_memory_system_factory(
        llm_model, llm_api_base, embedding_model
    )

    by_conversation: dict[str, list[LoCoMoInstance]] = defaultdict(list)
    for inst in instances:
        by_conversation[inst.conversation_id].append(inst)

    checkpoint_store = CheckpointStore(run_label, results_dir) if run_label else None
    prediction_store = PredictionStore(run_label, split, results_dir) if run_label else None

    progress_replay: Optional[ProgressReporter] = None
    progress_answer: Optional[ProgressReporter] = None
    if run_label:
        existing_checkpoints = {conv_id: checkpoint_store.load(conv_id) for conv_id in by_conversation}
        total_turns = sum(len(conv_instances[0].turns) for conv_instances in by_conversation.values())
        done_turns = sum(c.turns_processed for c in existing_checkpoints.values() if c is not None)
        done_conversations = sum(1 for c in existing_checkpoints.values() if c is not None and c.completed)
        print(
            f"[{run_label}] resuming: {done_conversations}/{len(by_conversation)} conversations "
            f"already replayed, {len(prediction_store)}/{len(instances)} questions already answered"
        )
        progress_replay = ProgressReporter(total=total_turns, label=f"{run_label}:replay", done=done_turns)
        progress_answer = ProgressReporter(
            total=len(instances), label=f"{run_label}:answer", done=len(prediction_store)
        )

    instance_results: list[InstanceResult] = []
    for conv_id, conv_instances in by_conversation.items():
        memory_system = factory(note_construction_prompt, evolution_prompt)
        replay_conversation(
            memory_system,
            conv_instances[0].turns,
            checkpoint_store=checkpoint_store,
            conversation_id=conv_id,
            progress=progress_replay,
        )

        for inst in conv_instances:
            cached = prediction_store.get(conv_id, inst.question) if prediction_store is not None else None
            if cached is not None:
                # Reuse the cached prediction (that's the expensive part --
                # an LLM call) but always rescore it fresh: metrics.py can
                # change (it did -- a tokenization bug meant real correct
                # answers scored 0.0) without invalidating every prediction
                # cache on disk.
                prediction = cached["prediction"]
                score = _score_one(prediction, inst)
            else:
                prediction = answer_question(memory_system, qa_prompt_template, inst.question, k=k)
                score = _score_one(prediction, inst)
                if prediction_store is not None:
                    prediction_store.append(
                        {
                            "conversation_id": conv_id,
                            "question": inst.question,
                            "category": inst.category,
                            "prediction": prediction,
                            "score": score,
                        }
                    )
                if progress_answer is not None:
                    progress_answer.tick(f"conv={conv_id}")

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
