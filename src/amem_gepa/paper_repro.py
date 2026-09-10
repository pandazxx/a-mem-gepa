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

import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPRO_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "agentic-memory-repro"


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
    """
    ensure_repro_repo_importable()

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
            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    conversation_tmp = "Speaker " + turn.speaker + "says : " + turn.text
                    agent.add_memory(conversation_tmp, time=turns.date_time)

            with open(memory_cache_file, "wb") as f:
                pickle.dump(agent.memory_system.memories, f)
            agent.memory_system.retriever.save(str(retriever_cache_file), str(retriever_cache_embeddings_file))
            eval_logger.info(f"[{sample_idx}] cached {len(agent.memory_system.memories)} memories")

        for qa in sample.qa:
            if int(qa.category) not in allow_categories:
                continue
            total_questions += 1
            category_counts[qa.category] += 1

            prediction, user_prompt, raw_context = agent.answer_question(qa.question, qa.category, qa.final_answer)
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
            all_metrics.append(metrics)
            all_categories.append(qa.category)
            results.append(
                {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics,
                }
            )
            if total_questions % 25 == 0:
                eval_logger.info(f"Processed {total_questions} questions")

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
