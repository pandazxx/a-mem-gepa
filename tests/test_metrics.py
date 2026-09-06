from amem_gepa.metrics import (
    bootstrap_ci,
    rouge_l_f1,
    score_adversarial,
    score_answer,
    score_batch,
)


def test_rouge_l_f1_identical():
    assert rouge_l_f1("7 May 2023", "7 May 2023") == 1.0


def test_rouge_l_f1_disjoint():
    assert rouge_l_f1("completely unrelated words here", "totally different reference text") == 0.0


def test_rouge_l_f1_partial_overlap():
    score = rouge_l_f1("the cat sat on the mat", "the cat sat on a rug")
    assert 0.0 < score < 1.0


def test_rouge_l_f1_empty_prediction():
    assert rouge_l_f1("", "some reference") == 0.0


def test_score_adversarial_avoided_trap():
    assert score_adversarial("I don't have enough information to answer that.", "self-care is important") == 1.0


def test_score_adversarial_fell_for_trap():
    assert score_adversarial("self-care is important", "self-care is important") == 0.0


def test_score_answer_dispatches_by_category():
    assert score_answer("Paris", category=1, gold_answer="Paris") == 1.0
    assert score_answer("no info", category=5, adversarial_answer="Paris") == 1.0


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
    predictions = ["Paris", "wrong", "no info here"]
    categories = [1, 2, 5]
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
