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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPRO_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "agentic-memory-repro"

# Matches external/agentic-memory-repro/run_k_sweep.sh's own K_VALUES.
K_SWEEP_VALUES = (10, 15, 20, 25, 30, 35, 40, 45, 50)


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
    ensure_nltk_data()

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

    Resumable per-k: a k whose output file under `results_dir/k_sweep/`
    already exists is loaded from disk instead of re-running its
    QA-answering pass.
    """
    sweep_dir = results_dir / "k_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    eval_logger = logging.getLogger("amem_gepa.paper_repro")

    results_by_k: dict[int, dict] = {}
    for k in k_values:
        out_file = sweep_dir / f"results_k{k}.json"
        if out_file.exists():
            eval_logger.info(f"[k-sweep] k={k} already computed -- loading {out_file}")
            results_by_k[k] = json.loads(out_file.read_text())
            continue

        eval_logger.info(f"[k-sweep] running retrieve_k={k}")
        result = run_full_reproduction(
            dataset_path=dataset_path,
            backend=backend,
            model=model,
            retrieve_k=k,
            ratio=ratio,
            temperature_c5=temperature_c5,
            results_dir=results_dir,
        )
        out_file.write_text(json.dumps(result, indent=2))
        results_by_k[k] = result

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
