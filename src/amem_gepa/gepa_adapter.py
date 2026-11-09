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

import json
import logging
import os
import pickle
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gepa.core.adapter import EvaluationBatch

from amem_gepa.datasets.locomo import LoCoMoInstance, LoCoMoQuestionGroup
from amem_gepa.paper_repro import (
    _format_duration,
    _ollama_call_stats,
    _patch_ollama_chat,
    _ProgressTracker,
    _summarize_ollama_calls,
    ensure_nltk_data,
    ensure_repro_repo_importable,
)
from amem_gepa.repro_injection import (
    REQUIRED_PLACEHOLDERS,
    BuildTrace,
    LLMCallRecord,
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

# Run-level counters persisted to `stats_path` after every evaluate() so a
# resumed run (same --run-label, docs/decisions/0017) reports totals for the
# whole run, not just the latest process.
_STATS_FIELDS = (
    "builds_performed",
    "build_cache_hits",
    "builds_resumed",
    "llm_calls",
    "questions_answered",
    "metric_calls",
    "evaluations",
    "build_seconds",
    "qa_seconds",
)


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
        checkpoint_every_turns: int = 20,
        log_every: int = 10,
        max_metric_calls: int | None = None,
        stats_path: Path | None = None,
        ensure_nltk: bool = True,
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
        # docs/decisions/0017: an in-progress build checkpoints (memories +
        # retriever + LLM-call trace) every N turns so an interruption costs
        # at most N turns, not the whole conversation. 0 disables.
        self.checkpoint_every_turns = checkpoint_every_turns
        # Progress lines (count / avg latency / ETA) every N turns built or
        # questions answered.
        self.log_every = max(1, log_every)
        self.max_metric_calls = max_metric_calls
        self.stats_path = Path(stats_path) if stats_path else None
        # Run-level accounting (docs/decisions/0015: builds are the budget
        # currency; CLAUDE.md wants cost totals logged per run).
        self.builds_performed = 0
        self.build_cache_hits = 0
        self.builds_resumed = 0
        self.llm_calls = 0
        self.questions_answered = 0
        self.metric_calls = 0
        self.evaluations = 0
        self.build_seconds = 0.0
        self.qa_seconds = 0.0
        # Wall-clock carried over from a previous process of the same run.
        self.prior_elapsed_seconds = 0.0
        self._started = time.monotonic()
        if self.stats_path is not None and self.stats_path.exists():
            self._load_stats()
        if ensure_nltk:
            # Same pre-flight `just reproduce` does (paper_repro.ensure_nltk_data):
            # the vendored utils.calculate_metrics calls nltk.word_tokenize,
            # which needs 'punkt_tab' at call time. Without this the first
            # real smoke run died with a LookupError on the very first scored
            # question -- *after* a full memory build. Done here, before any
            # build, so a missing resource fails in seconds instead.
            ensure_nltk_data()
        if backend == "ollama":
            # Same monkeypatch `just reproduce` installs (docs/decisions/0016):
            # forces `think=False` so a reasoning-capable local model
            # (qwen3.5:9b) doesn't spend ~100x the tokens per call here that
            # it was measured not to need there. Without this, the GEPA arm
            # and the baseline it is compared against would run the same
            # model under different generation settings.
            _patch_ollama_chat()

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[ItemTrajectory, Any]:
        started = time.monotonic()
        before = (self.builds_performed, self.build_cache_hits, self.questions_answered)
        result = self._evaluate(batch, candidate, capture_traces)
        self.metric_calls += len(batch)
        self.evaluations += 1
        logger.info(
            "[eval #%d] %d items: %d builds, %d cache hits, %d questions in %s | %s",
            self.evaluations,
            len(batch),
            self.builds_performed - before[0],
            self.build_cache_hits - before[1],
            self.questions_answered - before[2],
            _format_duration(time.monotonic() - started),
            self.run_progress_line(),
        )
        self._save_stats()
        return result

    def _evaluate(
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
        # Keyed by (conversation, turn count), not conversation alone: a
        # truncated smoke instance and a full instance of the same
        # conversation are different memory states and must not share a
        # build (or a cache entry -- see _cache_paths).
        agents: dict[tuple[str, int], tuple] = {}
        outputs: list[Any] = []
        scores: list[float] = []
        trajectories: list[ItemTrajectory] = []
        total_questions = sum(
            len(item.questions) if isinstance(item, LoCoMoQuestionGroup) else 1 for item in batch
        )
        qa_tracker = _ProgressTracker(logger, f"[eval #{self.evaluations + 1}] QA", total_questions)

        for item in batch:
            conv_id = self._conv_id(item)
            turns = self._turns(item)
            agent_key = (conv_id, len(turns))
            if agent_key not in agents:
                agents[agent_key] = self._build_or_load(candidate, cand_hash, conv_id, turns)
            agent, build_trace, cache_hit, build_error = agents[agent_key]

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
                q_traces = []
                for q in questions:
                    q_started = time.monotonic()
                    q_traces.append(self._answer_and_score(agent, q))
                    q_elapsed = time.monotonic() - q_started
                    qa_tracker.record(q_elapsed, cached=False)
                    self.qa_seconds += q_elapsed
                    self.questions_answered += 1
                    if qa_tracker.completed_units % self.log_every == 0 or (
                        qa_tracker.completed_units == total_questions
                    ):
                        logger.info(qa_tracker.progress_line(extra=f"conv={conv_id}"))

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

    def _cache_paths(self, cand_hash: str, conv_id: str, n_turns: int) -> dict[str, Path]:
        # n_turns in the path keeps truncated smoke builds (docs/decisions/
        # 0015's smoke addendum) from colliding with full builds of the
        # same conversation under the same candidate.
        base = self.cache_dir / cand_hash / f"{conv_id}_t{n_turns}"
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
        paths = self._cache_paths(cand_hash, conv_id, len(turns))
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
        partial = self._partial_paths(paths["dir"])
        start_turn = 0
        if partial["state"].exists():
            restored = self._restore_partial(agent, partial, conv_id, len(turns))
            if restored is not None:
                start_turn, prior_records = restored
                # Pre-interruption construction/evolution calls stay in the
                # trace, so reflection over this build still sees them.
                agent.memory_system.llm_controller.llm.records = list(prior_records)
                self.builds_resumed += 1
        records_before = len(agent.memory_system.llm_controller.llm.records)

        tracker = _ProgressTracker(
            logger, f"[build {conv_id} cand={cand_hash[:8]}] memory-build", len(turns)
        )
        for _ in range(start_turn):
            tracker.record(0.0, cached=True)
        build_started = time.monotonic()
        try:
            with injected_prompts(candidate):
                for idx in range(start_turn, len(turns)):
                    turn = turns[idx]
                    # Exact content format from test_advanced_robust.py's
                    # evaluate_dataset (including the missing space) -- the
                    # baseline numbers were produced with it, so the GEPA
                    # arm must match.
                    content = "Speaker " + turn.speaker + "says : " + turn.text
                    _ollama_call_stats.clear()
                    turn_started = time.monotonic()
                    agent.add_memory(content, time=turn.date_time)
                    elapsed = time.monotonic() - turn_started
                    tracker.record(elapsed, cached=False)
                    done = idx + 1
                    ollama_summary = _summarize_ollama_calls(list(_ollama_call_stats))
                    if tracker.is_outlier(elapsed):
                        logger.warning(
                            "[%s] slow memory-build turn %d: %.1fs (running avg %.1fs)%s",
                            conv_id,
                            done,
                            elapsed,
                            tracker.avg_latency,
                            f" -- {ollama_summary}" if ollama_summary else "",
                        )
                    if done % self.log_every == 0 or done == len(turns):
                        logger.info(tracker.progress_line(extra=ollama_summary))
                    checkpoint_due = (
                        self.checkpoint_every_turns
                        and done < len(turns)
                        and done % self.checkpoint_every_turns == 0
                    )
                    if checkpoint_due:
                        self._write_partial(agent, partial, conv_id, done)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[%s] memory build failed at turn %d/%d: %s",
                conv_id,
                tracker.completed_units,
                len(turns),
                e,
            )
            return None, None, False, str(e)

        build_trace.records = list(agent.memory_system.llm_controller.llm.records)
        self.builds_performed += 1
        self.build_seconds += time.monotonic() - build_started
        self.llm_calls += len(build_trace.records) - records_before

        try:
            paths["dir"].mkdir(parents=True, exist_ok=True)
            with open(paths["memories"], "wb") as f:
                pickle.dump(agent.memory_system.memories, f)
            agent.memory_system.retriever.save(str(paths["retriever"]), str(paths["retriever_embeddings"]))
            paths["trace"].write_text(build_trace.to_json())
            self._discard_partial(partial)
        except Exception as e:  # noqa: BLE001
            # A failed cache write must not fail the evaluation -- the
            # in-memory agent is still valid, we just pay the build again
            # next time.
            logger.warning("[%s] cache write failed: %s", conv_id, e)

        return agent, build_trace, False, None

    # ------------------------------------------------------------------
    # mid-build checkpoints (docs/decisions/0017)
    # ------------------------------------------------------------------

    @staticmethod
    def _partial_paths(build_dir: Path) -> dict[str, Path]:
        base = build_dir / "partial"
        return {
            "dir": base,
            "state": base / "state.pkl",
            "retriever": base / "retriever.pkl",
            "retriever_embeddings": base / "retriever_embeddings.npy",
        }

    def _write_partial(
        self, agent, partial: dict[str, Path], conv_id: str, turns_done: int
    ) -> None:
        """Snapshot an in-progress build. The memories, the turn count and
        the LLM-call records go in ONE pickle so they can't disagree with
        each other; the retriever is saved via the vendored code's own
        save() (same file format as the final cache) so the resumed build
        continues with exactly the retriever state it had, rather than one
        re-derived from the notes (which load_from_local_memory would
        build with a different document format). Every file is written to
        a temp name and os.replace()d, so a crash mid-write leaves the
        previous checkpoint intact; a crash between the replaces leaves a
        retriever/memories count mismatch that _restore_partial detects
        and discards."""
        try:
            partial["dir"].mkdir(parents=True, exist_ok=True)
            state = {
                "turns_done": turns_done,
                "evo_cnt": getattr(agent.memory_system, "evo_cnt", None),
                "memories": agent.memory_system.memories,
                "records": [r.to_dict() for r in agent.memory_system.llm_controller.llm.records],
            }
            tmp_state = partial["state"].with_suffix(".tmp")
            with open(tmp_state, "wb") as f:
                pickle.dump(state, f)
            tmp_retriever = partial["retriever"].with_suffix(".tmp")
            tmp_embeddings = partial["retriever_embeddings"].with_suffix(".tmp.npy")
            agent.memory_system.retriever.save(str(tmp_retriever), str(tmp_embeddings))
            os.replace(tmp_state, partial["state"])
            os.replace(tmp_retriever, partial["retriever"])
            if tmp_embeddings.exists():
                os.replace(tmp_embeddings, partial["retriever_embeddings"])
            logger.info("[%s] build checkpoint written at turn %d", conv_id, turns_done)
        except Exception as e:  # noqa: BLE001 -- a failed checkpoint must not fail the build
            logger.warning("[%s] build checkpoint at turn %d failed: %s", conv_id, turns_done, e)

    def _restore_partial(self, agent, partial: dict[str, Path], conv_id: str, n_turns: int):
        """Returns (turns_done, prior LLMCallRecords) after loading the
        checkpoint into `agent`, or None (with the checkpoint discarded) if
        it is unusable -- in which case the build simply starts over."""
        try:
            with open(partial["state"], "rb") as f:
                state = pickle.load(f)
            turns_done = int(state["turns_done"])
            memories = state["memories"]
            if not 0 < turns_done < n_turns:
                raise ValueError(f"turns_done={turns_done} is not inside (0, {n_turns})")
            retriever = agent.memory_system.retriever.load(
                str(partial["retriever"]), str(partial["retriever_embeddings"])
            )
            corpus = getattr(retriever, "corpus", None)
            if corpus is not None and len(corpus) != len(memories):
                raise ValueError(
                    f"retriever has {len(corpus)} documents but {len(memories)} memories"
                )
            agent.memory_system.memories = memories
            agent.memory_system.retriever = retriever
            if state.get("evo_cnt") is not None and hasattr(agent.memory_system, "evo_cnt"):
                agent.memory_system.evo_cnt = state["evo_cnt"]
            records = [LLMCallRecord(**r) for r in state.get("records", [])]
            logger.info(
                "[%s] resuming interrupted build at turn %d/%d (%d memories, %d LLM calls)",
                conv_id,
                turns_done,
                n_turns,
                len(memories),
                len(records),
            )
            return turns_done, records
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[%s] build checkpoint unusable (%s) -- rebuilding from scratch", conv_id, e
            )
            self._discard_partial(partial)
            return None

    @staticmethod
    def _discard_partial(partial: dict[str, Path]) -> None:
        for key in ("state", "retriever", "retriever_embeddings"):
            try:
                partial[key].unlink(missing_ok=True)
            except OSError:
                pass
        try:
            partial["dir"].rmdir()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # run-level progress + persisted accounting (docs/decisions/0017)
    # ------------------------------------------------------------------

    @property
    def elapsed_seconds(self) -> float:
        return self.prior_elapsed_seconds + (time.monotonic() - self._started)

    def eta_seconds(self) -> float | None:
        """Rough remaining wall-clock from the metric-call budget at the
        run's average pace so far. Honest about what it is: GEPA's cache-
        hit rate and acceptance rate change over a run, so this swings --
        it answers "days or hours?", not "when exactly?"."""
        if not self.max_metric_calls or self.metric_calls <= 0:
            return None
        remaining = max(self.max_metric_calls - self.metric_calls, 0)
        return self.elapsed_seconds / self.metric_calls * remaining

    def run_progress_line(self) -> str:
        bits = [f"run: builds={self.builds_performed}", f"cache_hits={self.build_cache_hits}"]
        if self.builds_resumed:
            bits.append(f"resumed_builds={self.builds_resumed}")
        if self.builds_performed:
            bits.append(f"avg_build={_format_duration(self.build_seconds / self.builds_performed)}")
        bits.append(f"questions={self.questions_answered}")
        if self.questions_answered:
            bits.append(f"avg_qa={self.qa_seconds / self.questions_answered:.1f}s")
        bits.append(f"llm_calls={self.llm_calls}")
        if self.max_metric_calls:
            bits.append(f"metric_calls={self.metric_calls}/{self.max_metric_calls}")
            eta = self.eta_seconds()
            if eta is not None:
                bits.append(f"ETA~{_format_duration(eta)} (rough, at current pace)")
        else:
            bits.append(f"metric_calls={self.metric_calls}")
        bits.append(f"elapsed={_format_duration(self.elapsed_seconds)}")
        return " | ".join(bits)

    def stats(self) -> dict:
        data = {field_name: getattr(self, field_name) for field_name in _STATS_FIELDS}
        data["elapsed_seconds"] = self.elapsed_seconds
        return data

    def _save_stats(self) -> None:
        if self.stats_path is None:
            return
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.stats_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.stats(), indent=2))
            os.replace(tmp, self.stats_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not write run stats to %s: %s", self.stats_path, e)

    def _load_stats(self) -> None:
        try:
            data = json.loads(self.stats_path.read_text())
            for field_name in _STATS_FIELDS:
                current = getattr(self, field_name)
                setattr(self, field_name, type(current)(data.get(field_name, current)))
            self.prior_elapsed_seconds = float(data.get("elapsed_seconds", 0.0))
            logger.info(
                "resuming run accounting from %s: %s", self.stats_path, self.run_progress_line()
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "could not read run stats from %s (%s) -- counting from zero", self.stats_path, e
            )

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
