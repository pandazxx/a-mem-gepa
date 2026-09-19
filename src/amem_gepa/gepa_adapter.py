"""GEPAAdapter optimizing the paper-reproduction pipeline's four memory
prompts (docs/decisions/0014, superseding 0002's two-prompt/library-pipeline
plan; docs/decisions/0015 for the training method).

Candidate components (see repro_injection.COMPONENT_TO_CONSTANT):
    note_construction, evolution_decision, evolution_strengthen,
    evolution_update_neighbors

The system under test is external/agentic-memory-repro's own code
(RobustAdvancedMemAgent), with the candidate's prompts swapped in at runtime
via repro_injection.injected_prompts -- retrieval (generate_query_llm), the
category-specific QA prompts, and all parsers stay frozen, so a score
difference between candidates is attributable to memory management only
(docs/decisions/0006's isolation argument, extended in 0014).

Cost model (docs/decisions/0015): the unit of cost is one memory *build* --
replaying a whole conversation through the candidate's prompts (~45 min /
~$0.35 on GPT-4o-mini, per docs/experiments/002's calibration). Questions
against an already-built memory are ~3 orders of magnitude cheaper. evaluate()
therefore groups its batch by conversation (one build each) and caches builds
on disk keyed by (candidate hash, conversation id), so re-evaluating a
candidate -- Pareto re-checks, resumed runs, the parent side of a minibatch
comparison -- costs zero builds.

Interface confirmed against the installed gepa package (gepa/core/adapter.py,
a Protocol): evaluate() returns EvaluationBatch(outputs, scores,
trajectories); scores are summed for minibatch acceptance and averaged over
the valset for Pareto tracking. Not thread-safe: injected_prompts patches
module-level state in the vendored code (GEPA's engine evaluates
sequentially, so this doesn't bite in practice).
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch

from amem_gepa.datasets.locomo import LoCoMoInstance, LoCoMoQuestionGroup
from amem_gepa.paper_repro import ensure_repro_repo_importable
from amem_gepa.repro_injection import (
    REQUIRED_PLACEHOLDERS,
    BuildTrace,
    RecordingLLM,
    candidate_hash,
    injected_prompts,
    validate_candidate,
)

logger = logging.getLogger("amem_gepa.gepa_adapter")

DataInst = LoCoMoInstance | LoCoMoQuestionGroup

# Below this per-question score a question counts as a failure for
# reflective-dataset purposes. 0.5 rather than "any imperfection": token-F1
# in the 0.5-1.0 range is usually a phrasing gap the frozen QA prompt owns,
# not a memory-management failure worth reflecting on.
FAILURE_THRESHOLD = 0.5

# The exact option string test_advanced_robust.py's category-5 multiple
# choice offers alongside the trap answer.
REFUSAL_OPTION = "not mentioned in the conversation"


@dataclass
class QuestionTrace:
    question: str
    category: int
    category_label: str
    prediction: str
    reference: str
    score: float
    evidence_turn_texts: list[str] = field(default_factory=list)
    evidence_retrieved: list[bool] = field(default_factory=list)
    raw_context: str = ""
    error: str | None = None


@dataclass
class ItemTrajectory:
    conversation_id: str
    questions: list[QuestionTrace] = field(default_factory=list)
    build_records: list = field(default_factory=list)  # list[LLMCallRecord]
    build_cache_hit: bool = False
    error: str | None = None
    validation_errors: list[str] = field(default_factory=list)


class AMemGEPAAdapter:
    """GEPAAdapter[DataInst, ItemTrajectory, RolloutOutput] over the
    paper-repro pipeline. Train instances are LoCoMoQuestionGroups (one
    build amortized over ~15 stratified questions); val instances are
    individual LoCoMoInstances (per-instance Pareto granularity) -- both
    are grouped by conversation here, so a full-valset evaluation still
    costs one build per val conversation."""

    def __init__(
        self,
        backend: str,
        model: str,
        retrieve_k: int = 10,
        temperature_c5: float = 0.5,
        cache_dir: Path = Path("results/gepa_cache"),
        adversarial_scoring: str = "refusal",
        max_reflection_examples: int = 5,
    ):
        if adversarial_scoring not in ("refusal", "paper"):
            raise ValueError(f"adversarial_scoring must be 'refusal' or 'paper', got {adversarial_scoring!r}")
        self.backend = backend
        self.model = model
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.cache_dir = Path(cache_dir)
        self.adversarial_scoring = adversarial_scoring
        self.max_reflection_examples = max_reflection_examples
        # Run-level accounting (docs/decisions/0015: builds are the budget
        # currency; CLAUDE.md wants cost totals logged per run).
        self.builds_performed = 0
        self.build_cache_hits = 0
        self.llm_calls = 0

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ItemTrajectory, Any]:
        problems = validate_candidate(candidate)
        if problems:
            # Never spend a build on a candidate that would crash at
            # .format() time -- score 0 with the problems as the trace, so
            # reflection can repair the template instead of the run dying.
            logger.warning("Invalid candidate (%d problems), scoring 0: %s", len(problems), problems)
            trajectories = [
                ItemTrajectory(conversation_id=self._conv_id(item), validation_errors=problems) for item in batch
            ]
            return EvaluationBatch(
                outputs=["" for _ in batch],
                scores=[0.0 for _ in batch],
                trajectories=trajectories if capture_traces else None,
                num_metric_calls=len(batch),
            )

        cand_hash = candidate_hash(candidate)
        agents: dict[str, tuple] = {}  # conversation_id -> (agent, BuildTrace|None, cache_hit, error)
        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[ItemTrajectory] = []

        for item in batch:
            conv_id = self._conv_id(item)
            if conv_id not in agents:
                agents[conv_id] = self._build_or_load(candidate, cand_hash, conv_id, self._turns(item))
            agent, build_trace, cache_hit, build_error = agents[conv_id]

            traj = ItemTrajectory(
                conversation_id=conv_id,
                build_records=build_trace.records if build_trace else [],
                build_cache_hit=cache_hit,
                error=build_error,
            )
            questions = item.questions if isinstance(item, LoCoMoQuestionGroup) else [item]

            if build_error is not None:
                q_traces = [
                    QuestionTrace(
                        question=q.question,
                        category=q.category,
                        category_label=q.category_label,
                        prediction="",
                        reference=self._reference(q),
                        score=0.0,
                        error=f"memory build failed: {build_error}",
                    )
                    for q in questions
                ]
            else:
                q_traces = [self._answer_and_score(agent, q) for q in questions]

            traj.questions = q_traces
            item_scores = [qt.score for qt in q_traces]
            score = sum(item_scores) / len(item_scores) if item_scores else 0.0
            prediction_out = (
                [qt.prediction for qt in q_traces]
                if isinstance(item, LoCoMoQuestionGroup)
                else q_traces[0].prediction
            )
            outputs.append(prediction_out)
            scores.append(score)
            trajectories.append(traj)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            num_metric_calls=len(batch),
        )

    # ------------------------------------------------------------------
    # build / cache
    # ------------------------------------------------------------------

    def _conv_id(self, item: DataInst) -> str:
        return item.conversation_id

    def _turns(self, item: DataInst) -> list:
        return item.turns

    def _cache_paths(self, cand_hash: str, conv_id: str) -> dict[str, Path]:
        base = self.cache_dir / cand_hash / conv_id
        return {
            "dir": base,
            "memories": base / "memories.pkl",
            "retriever": base / "retriever.pkl",
            "retriever_embeddings": base / "retriever_embeddings.npy",
            "trace": base / "build_trace.json",
        }

    def _make_agent(self, candidate: dict[str, str]):
        ensure_repro_repo_importable()
        from test_advanced_robust import RobustAdvancedMemAgent

        agent = RobustAdvancedMemAgent(self.model, self.backend, self.retrieve_k, self.temperature_c5)
        # Wrap both controllers (memory building + QA answering go through
        # memory_system's; query-keyword generation goes through
        # retriever_llm's) so every call is counted and classifiable.
        agent.memory_system.llm_controller.llm = RecordingLLM(agent.memory_system.llm_controller.llm, candidate)
        agent.retriever_llm.llm = RecordingLLM(agent.retriever_llm.llm, candidate)
        return agent

    def _build_or_load(self, candidate: dict[str, str], cand_hash: str, conv_id: str, turns: list) -> tuple:
        """Returns (agent, BuildTrace|None, cache_hit, error|None). The
        BuildTrace is persisted at build time regardless of capture_traces:
        a later capture_traces=True evaluation of the same (candidate,
        conversation) hits the cache and would otherwise have no
        construction/evolution trace to reflect over -- rebuilding just to
        re-record would cost a full build."""
        paths = self._cache_paths(cand_hash, conv_id)
        try:
            agent = self._make_agent(candidate)
        except Exception as e:  # noqa: BLE001 -- backend down, submodule missing, ...
            logger.error("[%s] agent construction failed: %s", conv_id, e)
            return None, None, False, str(e)

        if paths["memories"].exists():
            try:
                with open(paths["memories"], "rb") as f:
                    cached_memories = pickle.load(f)
                agent.memory_system.memories = cached_memories
                if paths["retriever"].exists():
                    agent.memory_system.retriever = agent.memory_system.retriever.load(
                        str(paths["retriever"]), str(paths["retriever_embeddings"])
                    )
                else:
                    agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                        cached_memories, "all-MiniLM-L6-v2"
                    )
                build_trace = (
                    BuildTrace.from_json(paths["trace"].read_text()) if paths["trace"].exists() else None
                )
                self.build_cache_hits += 1
                logger.info("[%s] cache hit for candidate %s (%d memories)", conv_id, cand_hash, len(cached_memories))
                return agent, build_trace, True, None
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] cache load failed (%s) -- rebuilding", conv_id, e)

        build_trace = BuildTrace(conversation_id=conv_id, candidate_hash=cand_hash)
        try:
            with injected_prompts(candidate):
                for turn in turns:
                    # Exact content format from test_advanced_robust.py's
                    # evaluate_dataset (including the missing space) -- the
                    # baseline numbers were produced with it, so the GEPA
                    # arm must match.
                    content = "Speaker " + turn.speaker + "says : " + turn.text
                    agent.add_memory(content, time=turn.date_time)
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] memory build failed: %s", conv_id, e)
            return None, None, False, str(e)

        build_trace.records = list(agent.memory_system.llm_controller.llm.records)
        self.builds_performed += 1
        self.llm_calls += len(build_trace.records)

        try:
            paths["dir"].mkdir(parents=True, exist_ok=True)
            with open(paths["memories"], "wb") as f:
                pickle.dump(agent.memory_system.memories, f)
            agent.memory_system.retriever.save(str(paths["retriever"]), str(paths["retriever_embeddings"]))
            paths["trace"].write_text(build_trace.to_json())
        except Exception as e:  # noqa: BLE001
            # A failed cache write must not fail the evaluation -- the
            # in-memory agent is still valid, we just pay the build again
            # next time.
            logger.warning("[%s] cache write failed: %s", conv_id, e)

        return agent, build_trace, False, None

    # ------------------------------------------------------------------
    # answering + scoring
    # ------------------------------------------------------------------

    def _reference(self, q: LoCoMoInstance) -> str:
        if q.is_adversarial:
            return str(q.adversarial_answer or "")
        return "" if q.gold_answer is None else str(q.gold_answer)

    def _score(self, prediction: str, q: LoCoMoInstance, reference: str) -> float:
        ensure_repro_repo_importable()
        from utils import calculate_metrics

        if q.is_adversarial:
            if self.adversarial_scoring == "refusal":
                # The paper's stated intent for category 5 ("assess models'
                # ability to identify unanswerable queries"): correct
                # behavior is picking the refusal option, not the trap
                # (docs/decisions/0015; the sign ambiguity in the paper's
                # own scoring is documented in 0013 and left open there).
                return 1.0 if REFUSAL_OPTION in prediction.lower() else 0.0
            # "paper" mode: their calculate_metrics against the trap text,
            # exactly as their aggregation does it -- kept selectable for
            # investigating the sign ambiguity, not the training default.
            return float(calculate_metrics(prediction, reference)["f1"]) if reference else 0.0

        if not reference:
            return 0.0
        return float(calculate_metrics(prediction, reference)["f1"])

    def _answer_and_score(self, agent, q: LoCoMoInstance) -> QuestionTrace:
        ensure_repro_repo_importable()
        from llm_text_parsers import parse_plain_text_answer

        reference = self._reference(q)
        qa_llm_calls_before = len(agent.memory_system.llm_controller.llm.records) + len(
            agent.retriever_llm.llm.records
        )
        try:
            raw_prediction, _user_prompt, raw_context = agent.answer_question(q.question, q.category, reference)
            prediction = parse_plain_text_answer(raw_prediction)
            error = None
        except Exception as e:  # noqa: BLE001
            prediction, raw_context, error = "", "", str(e)
        self.llm_calls += (
            len(agent.memory_system.llm_controller.llm.records)
            + len(agent.retriever_llm.llm.records)
            - qa_llm_calls_before
        )

        score = 0.0 if error else self._score(prediction, q, reference)

        evidence_texts = self._evidence_turn_texts(q)
        evidence_retrieved = [self._text_in_context(t, raw_context) for t in evidence_texts]

        return QuestionTrace(
            question=q.question,
            category=q.category,
            category_label=q.category_label,
            prediction=prediction,
            reference=reference,
            score=score,
            evidence_turn_texts=evidence_texts,
            evidence_retrieved=evidence_retrieved,
            raw_context=raw_context[:4000],
            error=error,
        )

    @staticmethod
    def _evidence_turn_texts(q: LoCoMoInstance) -> list[str]:
        if not q.evidence:
            return []
        by_dia = {t.dia_id: t.text for t in q.turns}
        return [by_dia[d] for d in q.evidence if d in by_dia]

    @staticmethod
    def _text_in_context(turn_text: str, context: str) -> bool:
        """Whether an evidence turn made it into the retrieved context.
        Substring on a normalized slice rather than equality: the context
        embeds turn text inside 'memory content: Speaker Xsays : ...' lines."""
        needle = " ".join(turn_text.split()).lower()[:80]
        return bool(needle) and needle in " ".join(context.split()).lower()

    # ------------------------------------------------------------------
    # make_reflective_dataset
    # ------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[ItemTrajectory, Any],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = eval_batch.trajectories or []

        # A candidate that failed placeholder validation never ran -- the
        # only useful reflection is "fix the template".
        validation_errors = next((t.validation_errors for t in trajectories if t.validation_errors), None)
        if validation_errors:
            return {
                component: [
                    {
                        "Inputs": {"current_prompt_template": candidate.get(component, "")},
                        "Generated Outputs": "(not executed -- template invalid)",
                        "Feedback": "The prompt template is invalid and scored 0 without running: "
                        + "; ".join(validation_errors)
                        + ". "
                        + self._placeholder_constraint(component),
                    }
                ]
                for component in components_to_update
            }

        dataset: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            if component == "note_construction":
                examples = self._note_construction_examples(trajectories)
            else:
                examples = self._evolution_examples(component, trajectories)
            if not examples:
                examples = [self._aggregate_example(component, trajectories)]
            for ex in examples:
                ex["Feedback"] += " " + self._placeholder_constraint(component)
            dataset[component] = examples[: self.max_reflection_examples]
        return dataset

    @staticmethod
    def _placeholder_constraint(component: str) -> str:
        required = sorted("{" + f + "}" for f in REQUIRED_PLACEHOLDERS.get(component, frozenset()))
        return (
            f"CONSTRAINT: the rewritten prompt is a Python str.format template and must contain exactly "
            f"these placeholders: {', '.join(required)} -- no other placeholders, and any literal brace "
            "must be doubled ('{{' / '}}'). It must also keep an explicitly parseable output format, since "
            "the response is consumed by a fixed plain-text parser."
        )

    def _failed_and_passed(self, trajectories: list[ItemTrajectory]) -> tuple[list, list]:
        failed, passed = [], []
        for traj in trajectories:
            for qt in traj.questions:
                (failed if qt.score < FAILURE_THRESHOLD else passed).append((traj, qt))
        return failed, passed

    @staticmethod
    def _find_records(traj: ItemTrajectory, kind: str, containing: str | None = None) -> list:
        needle = " ".join(containing.split()).lower()[:60] if containing else None
        out = []
        for record in traj.build_records:
            if record.kind != kind:
                continue
            if needle and needle not in " ".join(record.prompt.split()).lower():
                continue
            out.append(record)
        return out

    def _note_construction_examples(self, trajectories: list[ItemTrajectory]) -> list[dict[str, Any]]:
        """Failures attributed to note construction: the note's content IS
        the raw turn (the pipeline stores it verbatim), so what this prompt
        controls is *retrievability* -- keywords/context/tags are embedded
        alongside the content and are what similarity search actually
        matches on. A failed question whose evidence turn never showed up
        in the retrieved context points here."""
        examples: list[dict[str, Any]] = []
        failed, _ = self._failed_and_passed(trajectories)
        for traj, qt in failed:
            for turn_text, retrieved in zip(qt.evidence_turn_texts, qt.evidence_retrieved):
                records = self._find_records(traj, "note_construction", containing=turn_text)
                if not records:
                    continue
                if retrieved:
                    feedback = (
                        f"The evidence turn WAS retrieved into the QA context, yet the final answer "
                        f"was wrong (predicted {qt.prediction!r}, gold {qt.reference!r}, "
                        f"score {qt.score:.2f}). The extracted context/keywords may be diluting or "
                        "misrepresenting what the turn actually establishes."
                    )
                else:
                    feedback = (
                        f"A later {qt.category_label} question ({qt.question!r}, gold {qt.reference!r}) "
                        f"FAILED (predicted {qt.prediction!r}, score {qt.score:.2f}) and this evidence "
                        "turn was NOT among the retrieved memories. The keywords/context/tags produced "
                        "below are what retrieval embeds and matches against -- they did not surface "
                        "this turn for that question. Dates/times, entities, and relations the question "
                        "hinges on need to be recoverable from them."
                    )
                examples.append(
                    {
                        "Inputs": {
                            "turn_content (the {content} input)": turn_text,
                            "question_later_asked": qt.question,
                        },
                        "Generated Outputs": records[0].response,
                        "Feedback": feedback,
                    }
                )
                if len(examples) >= self.max_reflection_examples:
                    return examples
        return examples

    def _evolution_examples(self, component: str, trajectories: list[ItemTrajectory]) -> list[dict[str, Any]]:
        """Failures attributed to the evolution steps: links decide
        neighborhood expansion during retrieval (find_related_memories_raw
        follows note.links), and neighbor context/tag rewrites change what
        later retrieval embeds. An evolution call that touched a failed
        question's evidence turn is the most attributable example we can
        offer; failing that, any calls of this component paired with
        aggregate failure context."""
        examples: list[dict[str, Any]] = []
        failed, _ = self._failed_and_passed(trajectories)
        for traj, qt in failed:
            for turn_text, retrieved in zip(qt.evidence_turn_texts, qt.evidence_retrieved):
                for record in self._find_records(traj, component, containing=turn_text):
                    status = "was retrieved but the answer was still wrong" if retrieved else (
                        "was NOT among the retrieved memories for that question"
                    )
                    examples.append(
                        {
                            "Inputs": {"formatted_prompt_excerpt": record.prompt[:2500]},
                            "Generated Outputs": record.response,
                            "Feedback": (
                                f"This evolution step processed a turn that a later {qt.category_label} "
                                f"question depended on ({qt.question!r}, gold {qt.reference!r}); the "
                                f"evidence {status}, and the question scored {qt.score:.2f} "
                                f"(predicted {qt.prediction!r}). Links created here drive neighborhood "
                                "expansion at retrieval time, and tag/context updates change what "
                                "retrieval embeds -- this decision did not make the memory reachable "
                                "when it mattered."
                            ),
                        }
                    )
                    if len(examples) >= self.max_reflection_examples:
                        return examples
        return examples

    def _aggregate_example(self, component: str, trajectories: list[ItemTrajectory]) -> dict[str, Any]:
        """Fallback when no per-call attribution was possible (e.g. the
        component never fired on an evidence turn, or traces came back
        empty): one aggregate example so GEPA's proposer always has
        something concrete, per-category scores included so scarce
        categories stay visible (docs/decisions/0004)."""
        all_q = [qt for traj in trajectories for qt in traj.questions]
        by_cat: dict[str, list[float]] = {}
        for qt in all_q:
            by_cat.setdefault(qt.category_label, []).append(qt.score)
        cat_summary = ", ".join(
            f"{label}: {sum(scores) / len(scores):.2f} (n={len(scores)})" for label, scores in sorted(by_cat.items())
        )
        worst = sorted(all_q, key=lambda qt: qt.score)[:3]
        worst_summary = "; ".join(
            f"{qt.category_label} {qt.question!r} -> predicted {qt.prediction!r}, gold {qt.reference!r}"
            for qt in worst
        )
        fired = sum(len(self._find_records(traj, component)) for traj in trajectories)
        return {
            "Inputs": {"note": f"aggregate rollout summary ({len(all_q)} questions)"},
            "Generated Outputs": f"this component produced {fired} LLM calls in the traced builds",
            "Feedback": (
                f"No single call could be attributed to a specific failure. Mean score per category: "
                f"{cat_summary}. Worst questions: {worst_summary}. Consider whether this prompt's "
                "decisions (what to extract/link/update) preserve the information those questions needed."
            ),
        }
