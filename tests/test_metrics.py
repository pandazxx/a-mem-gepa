from amem_gepa.metrics import (
    bootstrap_ci,
    f1_multi_answer,
    f1_score,
    normalize_answer,
    rouge_l_f1,
    score_adversarial,
    score_answer,
    score_batch,
)


def test_normalize_answer_strips_commas_punctuation_and_articles():
    assert normalize_answer("The, quick! Fox.") == "quick fox"


def test_normalize_answer_strips_and_too():
    # Not a typo -- the paper's regex is \b(a|an|the|and)\b, "and" included.
    assert normalize_answer("pottery and camping") == "pottery camping"


def test_f1_score_exact_match():
    assert f1_score("Sweden", "Sweden") == 1.0


def test_f1_score_uses_stemming():
    # "painted"/"paints" both stem to "paint" -- credit even without an
    # exact surface match, matching the paper's Porter-stemmed comparison.
    assert f1_score("she paints often", "she painted yesterday") > 0.0


def test_f1_score_no_overlap():
    assert f1_score("completely unrelated", "totally different") == 0.0


def test_f1_score_ignores_trailing_punctuation():
    """Real example from a Llama 3.2:1b baseline run: this used to score
    0.0 under whitespace-only tokenization despite containing the right
    word, because "transgender." (with the period) didn't match
    "transgender". normalize_answer strips punctuation before comparing."""
    score = f1_score(
        "Based on the retrieved memories, Caroline's identity is transgender.",
        "Transgender woman",
    )
    assert score > 0.0


def test_f1_multi_answer_matches_best_sub_phrase_per_gold_item():
    """Real example: category-1 (multi-hop) gold answers are often several
    comma-separated facts drawn from different sessions."""
    score = f1_multi_answer(
        "She enjoys pottery classes and also goes camping most summers.",
        "pottery, camping, painting, swimming",
    )
    assert 0.0 < score < 1.0  # got 2 of 4, not all, not none


def test_f1_multi_answer_full_match():
    assert f1_multi_answer("pottery, camping, painting, swimming", "pottery, camping, painting, swimming") == 1.0


def test_score_adversarial_requires_exact_refusal_phrase():
    """Paper's method (task_eval/evaluation.py) is a literal substring
    check for two specific phrases -- not overlap with the trap answer, and
    not just "sounds like a refusal". A well-phrased but differently-worded
    refusal gets no credit; that's the paper's actual behavior, not a bug
    here."""
    assert score_adversarial("There is no information available about that.") == 1.0
    assert score_adversarial("That is not mentioned in the conversation.") == 1.0
    assert score_adversarial("I don't have enough context to answer that.") == 0.0
    assert score_adversarial("self-care is important") == 0.0


def test_score_answer_category_1_is_multi_hop_and_uses_multi_answer_f1():
    score = score_answer(
        "pottery, camping, painting, swimming", category=1, gold_answer="pottery, camping, painting, swimming"
    )
    assert score == 1.0


def test_score_answer_category_3_truncates_semicolon_variants():
    # Paper quirk: open-domain (category 3) gold answers may have
    # semicolon-separated variants; only the first is canonical.
    score = score_answer("Sweden", category=3, gold_answer="Sweden; Swedish city")
    assert score == 1.0


def test_score_answer_category_5_ignores_adversarial_answer_argument():
    # The paper's adversarial check never looks at the trap text at all.
    assert score_answer("not mentioned anywhere", category=5, adversarial_answer="Paris") == 1.0
    assert score_answer("Paris", category=5, adversarial_answer="Paris") == 0.0


def test_rouge_l_f1_still_available_as_secondary_signal():
    assert rouge_l_f1("7 May 2023", "7 May 2023") == 1.0
    assert rouge_l_f1("completely unrelated words here", "totally different reference text") == 0.0


def test_bootstrap_ci_tight_for_constant_scores():
    lo, hi = bootstrap_ci([1.0] * 20, n_resamples=500)
    assert lo == hi == 1.0


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_contains_mean():
    scores = [0.0, 1.0] * 10
    lo, hi = bootstrap_ci(scores, n_resamples=2000)
    assert lo <= 0.5 <= hi


def test_score_batch_per_category_and_aggregate():
    predictions = ["Paris", "wrong", "not mentioned in the conversation"]
    categories = [4, 2, 5]  # 4 = single_hop, 2 = temporal, 5 = adversarial
    gold = ["Paris", "7 May 2023", None]
    adversarial = [None, None, "the trap answer"]

    result = score_batch(predictions, categories, gold, adversarial, n_resamples=200)

    assert "aggregate" in result
    assert result["aggregate"].n == 3
    assert "single_hop" in result
    assert result["single_hop"].n == 1
    assert result["single_hop"].mean == 1.0
    assert "adversarial" in result
    assert result["adversarial"].mean == 1.0
