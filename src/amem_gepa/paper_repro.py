"""Faithful, end-to-end reproduction of the A-MEM paper's LoCoMo benchmark
(docs/decisions/0013), using the authors' own reproduction code
(`WujiangXu/AgenticMemory`, vendored read-only at
external/agentic-memory-repro/) rather than our own approximation.

Why a separate pipeline from evaluate.py/amem_adapter.py: that pipeline
wraps `agiresearch/A-mem`, the general-purpose *library* -- both its own
README and this repo's README say to use *this* repo instead to reproduce
the paper's actual numbers. The two use different prompts (4 here: note
construction + a 3-step evolution flow, vs. 2 there), different retrieval
(LLM-generated query keywords here, not the raw question), different QA
prompts (including a multiple-choice mechanic for adversarial questions),
and different scoring (their exact_match/F1/ROUGE/BLEU/BERTScore/METEOR/
SBERT suite, not our paper-derived-from-a-different-repo F1). Mixing them
would produce a number that's comparable to neither.

This module ports `evaluate_dataset()` from their `test_advanced_robust.py`
almost verbatim -- everything that affects the actual science (prompts,
retrieval, evolution, scoring) calls straight into their code, unmodified.
The only thing changed is *where files go*: their version writes memory/
retriever caches and logs relative to their own module's `__file__`, which
would write into this project's read-only vendored submodule directory;
this version redirects all of that under results/paper_repro/ (gitignored,
per CLAUDE.md).
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPRO_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "agentic-memory-repro"

# Matches external/agentic-memory-repro/run_k_sweep.sh's own K_VALUES.
K_SWEEP_VALUES = (10, 15, 20, 25, 30, 35, 40, 45, 50)

# The paper's actual per-category retrieve_k for each model's headline
# Table 1 number (Appendix A.5, Table 8, read directly from the paper --
# see docs/decisions/0013). Categories are the same integers as
# datasets.locomo.CATEGORY_LABELS (1=multi_hop, 2=temporal, 3=open_domain,
# 4=single_hop, 5=adversarial). Most models use k=10 everywhere ("For
# models that have already achieved SOTA performance with k=10, we
# maintain this value without further tuning" -- the paper's own words);
# only GPT-4o-mini/GPT-4o are tuned away from k=10 on every category.
PAPER_CATEGORY_K = {
    "gpt-4o-mini": {1: 40, 2: 40, 3: 50, 4: 50, 5: 40},
    "gpt-4o": {1: 40, 2: 40, 3: 50, 4: 50, 5: 40},
    "qwen2.5-1.5b": {1: 10, 2: 10, 3: 10, 4: 10, 5: 10},
    "qwen2.5-3b": {1: 10, 2: 10, 3: 50, 4: 10, 5: 10},
    "llama3.2-1b": {1: 10, 2: 10, 3: 10, 4: 10, 5: 10},
    "llama3.2-3b": {1: 10, 2: 20, 3: 10, 4: 10, 5: 10},
}


def ensure_repro_repo_importable() -> None:
    """Puts external/agentic-memory-repro on sys.path so its sibling-import
    modules (memory_layer_robust, llm_text_parsers, load_dataset, utils,
    test_advanced_robust) can be imported directly. Idempotent.

    Checks for a real file inside the directory, not just the directory's
    own existence -- an uninitialized git submodule leaves an *empty*
    directory in place (this project hit exactly this with external/a-mem
    early on: `REPRO_DIR.exists()` is true either way, so that check alone
    would pass and only fail later with a confusing bare ModuleNotFoundError
    from deep inside the `from test_advanced_robust import ...` line).
    """
    marker_file = REPRO_DIR / "test_advanced_robust.py"
    if not marker_file.exists():
        raise FileNotFoundError(
            f"{marker_file} not found -- external/agentic-memory-repro looks "
            "uninitialized (an empty git submodule directory, not a missing "
            "one, exists either way). Run `git submodule update --init --recursive` "
            "(see docs/decisions/0013)."
        )
    repro_dir_str = str(REPRO_DIR)
    if repro_dir_str not in sys.path:
        sys.path.insert(0, repro_dir_str)


def ensure_nltk_data() -> None:
    """Downloads the NLTK data external/agentic-memory-repro's own code
    actually needs at runtime, working around a version gap in their
    (read-only, not ours to edit) download logic: both
    test_advanced_robust.py and utils.py check for/download only 'punkt'
    and 'wordnet', which was correct when they were written, but NLTK
    3.8.2+ split punkt's tokenizer data into a separate 'punkt_tab'
    resource that word_tokenize() needs at call time -- their own
    pre-flight check doesn't catch this, so it surfaces later as a bare
    LookupError deep inside nltk's tokenizer, on the very first QA-answer
    scoring call. `nltk.download` is idempotent (a no-op if already
    present), so calling this on every run is cheap."""
    import nltk

    for resource in ("punkt", "punkt_tab", "wordnet"):
        nltk.download(resource, quiet=True)


def _qa_progress_path(results_dir: Path, backend: str, model: str, retrieve_k: int) -> Path:
    """Per-(backend, model, retrieve_k) log of already-answered questions,
    used to resume a run_full_reproduction call interrupted mid
    QA-answering. Unlike memory-building (cached per-conversation, keyed
    on backend/model only), the QA-answering loop had zero intra-run
    checkpointing until this was added -- a kill partway through
    re-answered every question from scratch on rerun, wasting real spend
    against a paid backend (hit for real running GPT-4o-mini via
    OpenRouter). Filename follows the same cached_memories_{backend}_{model}
    convention -- a `/` in `model` (e.g. "openai/gpt-4o-mini") produces a
    nested directory via pathlib the same way it already does there."""
    return results_dir / f"qa_progress_{backend}_{model}_k{retrieve_k}.jsonl"


def _load_qa_progress(path: Path) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    progress: dict[tuple[int, int], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        progress[(record["sample_id"], record["q_idx"])] = record
    return progress


def _append_qa_progress(path: Path, record: dict) -> None:
    # Real bug hit on a live run: a `/` in `model` (e.g. "openai/gpt-4o-mini")
    # makes `path` land in a nested directory (see _qa_progress_path's
    # docstring) that nothing had created yet -- open(path, "a") doesn't
    # create parent directories, unlike the memory cache dir's own
    # mkdir(parents=True) a few lines above every use of it.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _format_duration(seconds: float) -> str:
    """Human-readable duration for log lines (ETAs here are routinely
    multi-day, per docs/decisions/0013's hardware notes, so plain
    seconds aren't useful)."""
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours}h{minutes}m"
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


class _ProgressTracker:
    """Pure observability: tracks elapsed wall-clock time per unit of real
    work (one memory-building turn, one QA-answering question) to log
    throughput and an ETA. Added after a real run -- even after a GPU
    upgrade that should have comfortably fit the model in VRAM -- still
    took 3+ days on a single k and hadn't finished, with zero visibility
    into whether it was actually progressing, stalled, or just slow
    throughout. Doesn't change what gets computed, cached, or returned --
    only what gets logged, wrapped around the existing calls.

    Cache hits (an already-answered question resumed from
    `_qa_progress_path`, or an already-swept k loaded from disk) are
    recorded separately and excluded from the latency average -- they're
    near-instant and would otherwise make the ETA look far rosier than
    the real, uncached work remaining actually is.
    """

    def __init__(self, logger: logging.Logger, label: str, total_units: int):
        self.logger = logger
        self.label = label
        self.total_units = total_units
        self.completed_units = 0
        self.cached_units = 0
        self.last_elapsed: float | None = None
        self._real_elapsed_total = 0.0
        self._real_call_count = 0
        self.start_time = time.monotonic()

    def record(self, elapsed: float, cached: bool) -> None:
        self.completed_units += 1
        if cached:
            self.cached_units += 1
            return
        self._real_elapsed_total += elapsed
        self._real_call_count += 1
        self.last_elapsed = elapsed

    @property
    def avg_latency(self) -> float | None:
        return self._real_elapsed_total / self._real_call_count if self._real_call_count else None

    @property
    def eta_seconds(self) -> float | None:
        avg = self.avg_latency
        if avg is None:
            return None
        return avg * (self.total_units - self.completed_units)

    def is_outlier(self, elapsed: float, factor: float = 3.0) -> bool:
        # Needs a handful of real calls first so one slow cold-start call
        # doesn't get flagged against an average of essentially itself.
        avg = self.avg_latency
        return avg is not None and self._real_call_count >= 5 and elapsed > avg * factor

    def progress_line(self, extra: str | None = None) -> str:
        bits = [f"{self.label}: {self.completed_units}/{self.total_units}"]
        if self.cached_units:
            bits.append(f"{self.cached_units} cached")
        if self.last_elapsed is not None:
            bits.append(f"last={self.last_elapsed:.1f}s")
        avg = self.avg_latency
        if avg is not None:
            # n=<count> makes the ETA's confidence self-evident -- a real
            # run showed the ETA swing from ~8h to >1d within a handful of
            # log lines, which is correct behavior (a running average over
            # very few real samples is naturally volatile), not a bug, but
            # wasn't obvious without seeing how many samples it was based on.
            bits.append(f"avg={avg:.1f}s(n={self._real_call_count})")
        eta = self.eta_seconds
        if eta is not None:
            bits.append(f"ETA={_format_duration(eta)}")
        bits.append(f"elapsed={_format_duration(time.monotonic() - self.start_time)}")
        if extra:
            bits.append(extra)
        return " | ".join(bits)


# ---------------------------------------------------------------------------
# Ollama per-call timing breakdown + thinking-mode override (backend="ollama" only)
# ---------------------------------------------------------------------------
#
# Added after a real diagnosis session (docs/decisions/0016): `_ProgressTracker`'s
# own `elapsed` only measures OUR wall-clock around `agent.answer_question()`,
# which doesn't say whether that time is prompt prefill, generation, or
# something else entirely. The vendored `RobustOllamaController.get_completion`
# (external/agentic-memory-repro/memory_layer_robust.py, read-only) calls
# `ollama.chat()` and keeps only `response["message"]["content"]`,
# discarding the detailed timing fields the Ollama API actually returns
# (`total_duration`/`load_duration`/`prompt_eval_count`/
# `prompt_eval_duration`/`eval_count`/`eval_duration`, all in
# nanoseconds -- confirmed against the installed `ollama` package's
# `ChatResponse`/`SubscriptableBaseModel`, which supports `.get()` safely
# even for an unset field).
#
# That logging led to the actual root cause (docs/decisions/0016): `ollama
# show qwen3.5:9b` reports `thinking: default true` -- the model generates
# an extended reasoning block before every answer unless told not to, and
# the vendored get_completion never passes Ollama's own `think` parameter
# (confirmed in the installed `ollama` package's `chat()` signature) to
# turn it off. A live side-by-side comparison on the same prompt: default
# (thinking on) used 1,115 eval tokens / 15.3s for a short keyword
# extraction; `think=False` used 11 eval tokens / 0.1s for a functionally
# identical answer. None of the paper's six originally-benchmarked models
# (GPT-4o-mini/GPT-4o/Qwen2.5-1.5b/3b/Llama-3.2-1b/3b) are reasoning
# models, so disabling thinking for qwen3.5:9b keeps this comparison
# faithful to the paper's own fast-instruction-following methodology
# rather than introducing a deviation.
#
# Rather than editing that vendored file, monkeypatch `ollama.chat` itself
# (the third-party package, not the submodule) to capture timing detail
# AND force `think=False` on the way through. `from ollama import chat`
# inside `get_completion` re-resolves `ollama.chat` on every call (it's a
# function-local import, not a module-load-time one), so patching the
# attribute before that line runs is picked up transparently.

_ollama_call_stats: list[dict] = []


def _ns_to_s(ns) -> float | None:
    return ns / 1e9 if ns is not None else None


def _patch_ollama_chat() -> None:
    """Idempotent -- safe to call on every run_full_reproduction call
    (e.g. once per k-sweep value). No-op the second time onward."""
    import ollama

    if getattr(ollama.chat, "_amem_gepa_instrumented", False):
        return

    original_chat = ollama.chat

    def _instrumented_chat(*args, **kwargs):
        # setdefault, not an unconditional override: the vendored code
        # never passes `think` today, so this is the only thing that
        # changes in practice, but a future/different caller that does
        # pass it explicitly has its choice respected rather than
        # silently fought.
        kwargs.setdefault("think", False)
        call_start = time.monotonic()
        response = original_chat(*args, **kwargs)
        wall_elapsed = time.monotonic() - call_start
        # content_preview/content_tail: token counts alone can only suggest
        # "this generated way more than a short structured extraction/
        # decision should need" -- the actual text directly confirms (or
        # rules out) *why*. A live run's head-only previews showed a
        # perfectly normal, short-looking start ("Keywords: Audrey, dogs,
        # games, park") for calls that still generated thousands of eval
        # tokens -- no `<think>` block, so whatever's happening keeps
        # generating *after* a reasonable-looking answer rather than
        # reasoning *before* it. The tail is what actually shows that:
        # trailing repetition/looping reads very differently from a
        # coherent, intentionally long continuation.
        message = response.get("message") or {}
        content = message.get("content") or ""
        _ollama_call_stats.append(
            {
                "wall_elapsed": wall_elapsed,
                "total_duration": _ns_to_s(response.get("total_duration")),
                "load_duration": _ns_to_s(response.get("load_duration")),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "prompt_eval_duration": _ns_to_s(response.get("prompt_eval_duration")),
                "eval_count": response.get("eval_count"),
                "eval_duration": _ns_to_s(response.get("eval_duration")),
                "content_len": len(content),
                "content_preview": content[:80].replace("\n", "\\n"),
                "content_tail": content[-80:].replace("\n", "\\n") if len(content) > 80 else "",
            }
        )
        return response

    _instrumented_chat._amem_gepa_instrumented = True
    ollama.chat = _instrumented_chat


def _summarize_ollama_calls(calls: list[dict]) -> str | None:
    """Compact one-line summary for a question's/turn's Ollama call(s) --
    `answer_question()` makes 2 (keyword generation, then the answer
    itself), memory-building's add_note makes 1-4 depending on whether
    evolution fires. Returns None (nothing to append) when empty, e.g. for
    a non-ollama backend, where the monkeypatch above is never installed
    so nothing is ever recorded."""
    if not calls:
        return None
    wall = sum(c["wall_elapsed"] for c in calls)
    reported = sum(c["total_duration"] or 0.0 for c in calls)
    prompt_tokens = sum(c["prompt_eval_count"] or 0 for c in calls)
    prompt_time = sum(c["prompt_eval_duration"] or 0.0 for c in calls)
    eval_tokens = sum(c["eval_count"] or 0 for c in calls)
    eval_time = sum(c["eval_duration"] or 0.0 for c in calls)
    unaccounted = wall - reported

    def _call_preview(i: int, c: dict) -> str:
        head = c.get("content_preview")
        if not head:
            return ""
        length = c.get("content_len")
        bits = [f"[{i}]"]
        if length is not None:
            bits.append(f"(len={length})")
        bits.append(f'head="{head}"')
        tail = c.get("content_tail")
        if tail:
            bits.append(f'tail="{tail}"')
        return " ".join(bits)

    previews = " ".join(p for p in (_call_preview(i, c) for i, c in enumerate(calls)) if p)
    preview_suffix = f" previews: {previews}" if previews else ""
    return (
        f"ollama[{len(calls)} call(s)]: wall={wall:.1f}s reported={reported:.1f}s "
        f"prompt={prompt_tokens}tok/{prompt_time:.1f}s eval={eval_tokens}tok/{eval_time:.1f}s "
        f"unaccounted={unaccounted:.1f}s{preview_suffix}"
    )


def run_full_reproduction(
    dataset_path: Path,
    backend: str,
    model: str,
    retrieve_k: int = 10,
    ratio: float = 1.0,
    temperature_c5: float = 0.5,
    results_dir: Path = Path("results/paper_repro"),
) -> dict:
    """Ported from test_advanced_robust.py:evaluate_dataset, redirecting
    cache/log/output paths under `results_dir` instead of the vendored
    submodule's own directory. See module docstring for what's reused
    unmodified (everything except file paths) vs. what's different from
    evaluate.py's pipeline.

    QA-answering is resumable per-question via `_qa_progress_path` --
    added after a real GPT-4o-mini/OpenRouter run got killed partway
    through and lost everything, since only memory-building had any
    checkpointing before this.
    """
    ensure_repro_repo_importable()
    ensure_nltk_data()
    if backend == "ollama":
        _patch_ollama_chat()

    # Imported here, not at module level: importing test_advanced_robust
    # eagerly loads a SentenceTransformer model and does NLTK downloads at
    # import time -- fine for an actual run, but means every caller of this
    # module (including tests that only want e.g. path helpers) would pay
    # that cost otherwise.
    from test_advanced_robust import RobustAdvancedMemAgent
    from load_dataset import load_locomo_dataset
    from utils import calculate_metrics, aggregate_metrics
    from llm_text_parsers import parse_plain_text_answer

    eval_logger = logging.getLogger("amem_gepa.paper_repro")

    samples = load_locomo_dataset(str(dataset_path))
    eval_logger.info(f"Loaded {len(samples)} conversations from {dataset_path}")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} conversations ({ratio * 100:.1f}% of dataset)")

    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts: dict = defaultdict(int)

    memories_dir = results_dir / f"cached_memories_{backend}_{model}"
    memories_dir.mkdir(parents=True, exist_ok=True)
    allow_categories = [1, 2, 3, 4, 5]

    progress_path = _qa_progress_path(results_dir, backend, model, retrieve_k)
    qa_progress = _load_qa_progress(progress_path)
    if qa_progress:
        eval_logger.info(
            f"Resuming QA-answering at k={retrieve_k}: {len(qa_progress)} questions already answered"
        )

    qa_tracker = _ProgressTracker(eval_logger, f"QA-answering k={retrieve_k}", sum(len(s.qa) for s in samples))

    for sample_idx, sample in enumerate(samples):
        # RobustAdvancedMemAgent's own signature: (model, backend, retrieve_k,
        # temperature_c5, sglang_host=..., sglang_port=...) -- no api_base;
        # for backend="ollama" it goes through RobustOllamaController, which
        # uses the `ollama` package's own default host resolution.
        agent = RobustAdvancedMemAgent(model, backend, retrieve_k, temperature_c5)

        memory_cache_file = memories_dir / f"memory_cache_sample_{sample_idx}.pkl"
        retriever_cache_file = memories_dir / f"retriever_cache_sample_{sample_idx}.pkl"
        retriever_cache_embeddings_file = memories_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"

        if memory_cache_file.exists():
            eval_logger.info(f"[{sample_idx}] loading cached memories")
            with open(memory_cache_file, "rb") as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if retriever_cache_file.exists():
                agent.memory_system.retriever = agent.memory_system.retriever.load(
                    str(retriever_cache_file), str(retriever_cache_embeddings_file)
                )
            else:
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
                    cached_memories, "all-MiniLM-L6-v2"
                )
            eval_logger.info(f"[{sample_idx}] loaded {len(cached_memories)} memories")
        else:
            eval_logger.info(f"[{sample_idx}] no cache -- replaying conversation")
            total_turns_this_sample = sum(len(turns.turns) for _, turns in sample.conversation.sessions.items())
            build_tracker = _ProgressTracker(eval_logger, f"[{sample_idx}] memory-build", total_turns_this_sample)
            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                    _ollama_call_stats.clear()
                    call_start = time.monotonic()
                    agent.add_memory(conversation_tmp, time=turns.date_time)
                    elapsed = time.monotonic() - call_start
                    ollama_calls = list(_ollama_call_stats)
                    build_tracker.record(elapsed, cached=False)
                    if build_tracker.is_outlier(elapsed):
                        extra = f" -- {_summarize_ollama_calls(ollama_calls)}" if ollama_calls else ""
                        eval_logger.warning(
                            f"[{sample_idx}] slow memory-build call: {elapsed:.1f}s "
                            f"(running avg {build_tracker.avg_latency:.1f}s) -- "
                            f"content_len={len(conversation_tmp)}{extra}"
                        )
                    eval_logger.info(build_tracker.progress_line(extra=_summarize_ollama_calls(ollama_calls)))

            with open(memory_cache_file, "wb") as f:
                pickle.dump(agent.memory_system.memories, f)
            agent.memory_system.retriever.save(str(retriever_cache_file), str(retriever_cache_embeddings_file))
            eval_logger.info(f"[{sample_idx}] cached {len(agent.memory_system.memories)} memories")

        for q_idx, qa in enumerate(sample.qa):
            if int(qa.category) not in allow_categories:
                continue
            total_questions += 1
            category_counts[qa.category] += 1

            cached = qa_progress.get((sample_idx, q_idx))
            ollama_calls: list[dict] = []
            if cached is not None:
                prediction = cached["prediction"]
                metrics = cached["metrics"]
                qa_tracker.record(0.0, cached=True)
            else:
                _ollama_call_stats.clear()
                call_start = time.monotonic()
                prediction, user_prompt, raw_context = agent.answer_question(
                    qa.question, qa.category, qa.final_answer
                )
                elapsed = time.monotonic() - call_start
                ollama_calls = list(_ollama_call_stats)
                prediction = parse_plain_text_answer(prediction)

                metrics = (
                    calculate_metrics(prediction, qa.final_answer)
                    if qa.final_answer
                    else {
                        "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                        "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                        "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0,
                    }
                )

                qa_tracker.record(elapsed, cached=False)
                if qa_tracker.is_outlier(elapsed):
                    # context_len is the diagnostic that matters most at high
                    # k: a bigger retrieved context means a bigger prompt on
                    # every single call, since there's no shared-prefix KV
                    # reuse across different questions' different top-k sets
                    # -- this is the field to check if slow calls correlate
                    # with large k rather than a stalled/misbehaving backend.
                    # ollama_summary adds the actual prefill/eval timing
                    # breakdown for backend="ollama" (see _patch_ollama_chat),
                    # since a live diagnosis found `elapsed` alone can't tell
                    # prefill-bound from eval-bound from genuinely-unaccounted-for
                    # time.
                    ollama_summary = _summarize_ollama_calls(ollama_calls)
                    extra = f" -- {ollama_summary}" if ollama_summary else ""
                    eval_logger.warning(
                        f"[{sample_idx}:{q_idx}] slow QA call: {elapsed:.1f}s "
                        f"(running avg {qa_tracker.avg_latency:.1f}s) -- "
                        f"category={qa.category} context_len={len(raw_context) if raw_context else 0} "
                        f"response_len={len(prediction) if prediction else 0}{extra}"
                    )

            all_metrics.append(metrics)
            all_categories.append(qa.category)
            record = {
                "sample_id": sample_idx,
                "q_idx": q_idx,
                "question": qa.question,
                "prediction": prediction,
                "reference": qa.final_answer,
                "category": qa.category,
                "metrics": metrics,
            }
            results.append(record)
            if cached is None:
                _append_qa_progress(progress_path, record)
            progress_extra = f"sample={sample_idx} cat={qa.category}"
            ollama_summary = _summarize_ollama_calls(ollama_calls)
            if ollama_summary:
                progress_extra = f"{progress_extra} | {ollama_summary}"
            eval_logger.info(qa_tracker.progress_line(extra=progress_extra))

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    final_results = {
        "model": model,
        "backend": backend,
        "retrieve_k": retrieve_k,
        "dataset": str(dataset_path),
        "generated_at": datetime.now().isoformat(),
        "total_questions": total_questions,
        "category_distribution": {str(cat): count for cat, count in category_counts.items()},
        "aggregate_metrics": aggregate_results,
        "individual_results": results,
    }
    return final_results


def run_k_sweep(
    dataset_path: Path,
    backend: str,
    model: str,
    k_values: tuple[int, ...] = K_SWEEP_VALUES,
    ratio: float = 1.0,
    temperature_c5: float = 0.5,
    results_dir: Path = Path("results/paper_repro"),
) -> dict[int, dict]:
    """Finds the best single, uniform `retrieve_k` for a fixed backend/model,
    sweeping the same range as the paper's own run_k_sweep.sh.

    Correction (2026-09-12), read directly from the paper (Appendix A.5,
    Table 8, see docs/decisions/0013): the paper's headline numbers do NOT
    use one swept k per model applied uniformly -- most models (including
    Llama-3.2-1b, this project's M2 model) use k=10 for every category,
    period. Only GPT-4o-mini/GPT-4o (k=40 for Multi Hop/Temporal/
    Adversarial, k=50 for Open Domain/Single Hop) and, to a lesser degree,
    Qwen2.5-3b/Llama-3.2-3b (one category each) get tuned away from k=10,
    and even then it's a *per-category* choice within one run, not "pick
    this model's single best global k". This function's one-k-for-the-
    whole-dataset design can find the best *uniform* k, which is a
    reasonable thing to know, but can't reconstruct the paper's actual
    per-category-spliced number by itself -- for that, run this at k=40
    and k=50 specifically (cheaper than the full sweep) and take Multi
    Hop/Temporal/Adversarial from the k=40 result, Open Domain/Single Hop
    from k=50.

    Cheap relative to a from-scratch run, NOT free: `run_full_reproduction`
    caches memories under `results_dir/cached_memories_{backend}_{model}/`
    keyed on backend/model only, not retrieve_k (confirmed by reading
    test_advanced_robust.py:evaluate_dataset, which their own
    run_k_sweep.sh relies on the same way) -- so every k here after the
    first reuses those memories rather than rebuilding them. What's NOT
    skipped is QA-answering: each k still re-answers all `n` questions (2
    LLM calls per question -- keyword generation, then the answer), so
    this is `len(k_values)`x the QA-answering cost of one `run_full_reproduction`
    call, not free.

    Resumable per-k (a k whose output file under `results_dir/k_sweep/`
    already exists is loaded from disk, no rerun at all) AND, within a
    still-incomplete k, resumable per-question via
    `run_full_reproduction`'s own QA-answering checkpoint -- a kill
    partway through one k's QA-answering no longer loses that k's
    progress on rerun.
    """
    sweep_dir = results_dir / "k_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    eval_logger = logging.getLogger("amem_gepa.paper_repro")

    sweep_tracker = _ProgressTracker(eval_logger, "[k-sweep]", len(k_values))
    results_by_k: dict[int, dict] = {}
    for k in k_values:
        # Real bug hit on a live run: this filename used to be just
        # f"results_k{k}.json" -- no backend/model in it at all. Running
        # the sweep for a second model reused the *first* model's k=40/50
        # result files verbatim (the "already computed" resumability check
        # below has no way to tell them apart), silently reporting the old
        # model's numbers as if they were the new model's, no error, no
        # warning. Matches _qa_progress_path's naming (backend_model_k) --
        # a "/" in `model` (e.g. "openai/gpt-4o-mini") produces a nested
        # directory via pathlib the same way it already does there.
        out_file = sweep_dir / f"results_{backend}_{model}_k{k}.json"
        if out_file.exists():
            eval_logger.info(f"[k-sweep] k={k} already computed for {backend}/{model} -- loading {out_file}")
            results_by_k[k] = json.loads(out_file.read_text())
            sweep_tracker.record(0.0, cached=True)
            continue

        eval_logger.info(
            f"[k-sweep] running retrieve_k={k} ({sweep_tracker.completed_units + 1}/{len(k_values)}) -- "
            f"see per-question logging below for this k's own progress/ETA"
        )
        k_start = time.monotonic()
        result = run_full_reproduction(
            dataset_path=dataset_path,
            backend=backend,
            model=model,
            retrieve_k=k,
            ratio=ratio,
            temperature_c5=temperature_c5,
            results_dir=results_dir,
        )
        k_elapsed = time.monotonic() - k_start
        sweep_tracker.record(k_elapsed, cached=False)
        # Same "/" in model" nested-directory consequence as
        # _append_qa_progress hit (d71f798) -- mkdir before writing.
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(result, indent=2))
        results_by_k[k] = result
        eval_logger.info(f"[k-sweep] k={k} finished in {_format_duration(k_elapsed)} -- {sweep_tracker.progress_line()}")

    return results_by_k


def format_k_sweep_summary(results_by_k: dict[int, dict]) -> str:
    """Overall F1/BLEU-1 per k (mirrors run_k_sweep.sh's own summary
    block), plus the per-category breakdown at the best overall-F1 k --
    docs/experiments/001 found the paper gap concentrated in specific
    categories (temporal/open_domain/adversarial), not spread evenly, so
    the best *overall* k could still look worse on any one category."""
    from amem_gepa.datasets.locomo import CATEGORY_LABELS

    ks = sorted(results_by_k)
    best_k = max(ks, key=lambda k: results_by_k[k]["aggregate_metrics"]["overall"]["f1"]["mean"])

    lines = [f"{'k':>4} {'overall F1':>10} {'overall BLEU-1':>14}", "-" * 32]
    for k in ks:
        overall = results_by_k[k]["aggregate_metrics"]["overall"]
        marker = "  <-- best overall F1" if k == best_k else ""
        lines.append(f"{k:>4} {overall['f1']['mean']:>10.4f} {overall['bleu1']['mean']:>14.4f}{marker}")

    lines.append("")
    lines.append(f"Per-category at best overall-F1 k={best_k}:")
    metrics = results_by_k[best_k]["aggregate_metrics"]
    for cat in sorted(int(c) for c in results_by_k[best_k]["category_distribution"]):
        stats = metrics.get(f"category_{cat}", {})
        f1 = stats.get("f1", {}).get("mean")
        bleu1 = stats.get("bleu1", {}).get("mean")
        n = stats.get("f1", {}).get("count", 0)
        label = CATEGORY_LABELS.get(cat, f"category_{cat}")
        lines.append(f"  {label:14s} F1={f1:.4f} BLEU-1={bleu1:.4f} n={n}")

    return "\n".join(lines)


def splice_per_category_results(results_by_k: dict[int, dict], category_k: dict[int, int]) -> dict:
    """Reconstructs the paper's actual per-category k selection (Table 8,
    PAPER_CATEGORY_K) from a set of run_k_sweep results. Neither
    run_full_reproduction nor run_k_sweep apply a different k per category
    within a single run -- each call is one uniform k end to end -- so
    matching the paper's real headline number for a model that (unlike
    Llama-3.2-1b) got per-category tuning means picking each category's
    stats from the k run specified for it in `category_k`, not from
    whichever single k scored highest overall.

    `results_by_k` must have an entry for every k value referenced in
    `category_k` (i.e. run_k_sweep with at least those k_values) or a
    KeyError is raised naming the missing k -- silently skipping a missing
    category would produce a spliced result that looks complete but is
    missing data, worse than failing loudly.

    Returns per-category {f1, bleu1, n, k} plus an "overall" entry that is
    the n-weighted mean across the spliced categories -- NOT a number the
    paper itself reports (docs/experiments/001 established the paper has
    no combined overall column, only per-category and a rank column across
    competing methods), computed here purely as a convenient single
    number; callers should label it as such, not as "the paper's overall".
    """
    spliced: dict = {}
    total_n = 0
    weighted_f1 = 0.0
    weighted_bleu = 0.0
    for cat, k in category_k.items():
        if k not in results_by_k:
            raise KeyError(
                f"category {cat} needs k={k}, but results_by_k has no entry for it "
                f"-- rerun run_k_sweep with k_values including {k}"
            )
        stats = results_by_k[k]["aggregate_metrics"][f"category_{cat}"]
        n = stats["f1"]["count"]
        spliced[cat] = {"f1": stats["f1"]["mean"], "bleu1": stats["bleu1"]["mean"], "n": n, "k": k}
        total_n += n
        weighted_f1 += stats["f1"]["mean"] * n
        weighted_bleu += stats["bleu1"]["mean"] * n

    if total_n:
        spliced["overall"] = {"f1": weighted_f1 / total_n, "bleu1": weighted_bleu / total_n, "n": total_n}
    return spliced


def format_spliced_summary(spliced: dict) -> str:
    """Table matching format_summary's layout, plus the k used per
    category (since unlike a single run_full_reproduction result, a
    spliced one mixes k values by design)."""
    from amem_gepa.datasets.locomo import CATEGORY_LABELS

    lines = [f"{'category':14s} {'k':>4s} {'F1':>8s} {'BLEU-1':>8s} {'n':>6s}"]
    for cat in sorted(c for c in spliced if c != "overall"):
        stats = spliced[cat]
        label = CATEGORY_LABELS.get(cat, f"category_{cat}")
        lines.append(f"{label:14s} {stats['k']:4d} {stats['f1']:8.4f} {stats['bleu1']:8.4f} {stats['n']:6d}")
    if "overall" in spliced:
        overall = spliced["overall"]
        lines.append(f"{'overall (n-weighted, not a paper number)':40s} {overall['f1']:8.4f} {overall['bleu1']:8.4f} {overall['n']:6d}")
    return "\n".join(lines)


def format_summary(final_results: dict) -> str:
    """Table matching the paper's own presentation (Multi Hop/Temporal/Open
    Domain/Single Hop/Adversarial x F1/BLEU) for a direct side-by-side
    comparison, pulling category names from datasets.locomo.CATEGORY_LABELS
    (verified against this exact repo's README, docs/decisions/0010)."""
    from amem_gepa.datasets.locomo import CATEGORY_LABELS

    lines = [f"model={final_results['model']} backend={final_results['backend']} "
             f"retrieve_k={final_results['retrieve_k']} n={final_results['total_questions']}", ""]
    lines.append(f"{'category':14s} {'F1':>8s} {'BLEU-1':>8s} {'n':>6s}")
    metrics = final_results["aggregate_metrics"]
    for cat in sorted(int(k) for k in final_results["category_distribution"]):
        stats = metrics.get(f"category_{cat}", {})
        f1 = stats.get("f1", {}).get("mean")
        bleu1 = stats.get("bleu1", {}).get("mean")
        n = stats.get("f1", {}).get("count", 0)
        label = CATEGORY_LABELS.get(cat, f"category_{cat}")
        lines.append(f"{label:14s} {f1:8.4f} {bleu1:8.4f} {n:6d}" if f1 is not None else f"{label:14s} {'--':>8s} {'--':>8s} {n:6d}")
    overall = metrics.get("overall", {})
    lines.append(
        f"{'overall':14s} {overall.get('f1', {}).get('mean', 0):8.4f} "
        f"{overall.get('bleu1', {}).get('mean', 0):8.4f} {final_results['total_questions']:6d}"
    )
    return "\n".join(lines)
