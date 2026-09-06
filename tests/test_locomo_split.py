from amem_gepa.datasets.locomo import (
    _category_balance_score,
    category_counts_by_conversation,
    make_split,
)


def _fake_conversations():
    # 10 conversations, skewed category counts (mirrors the real dataset's
    # imbalance -- category 3 is scarce, category 4 dominates) so the balance
    # search actually has to do work, not just split evenly.
    convs = []
    for i in range(10):
        qa = (
            [{"category": 1}] * (5 + i)
            + [{"category": 2}] * (6 + i)
            + [{"category": 3}] * 2
            + [{"category": 4}] * (15 + i)
            + [{"category": 5}] * (8 + i)
        )
        convs.append({"sample_id": f"conv-{i}", "qa": qa})
    return convs


def test_make_split_sizes():
    split = make_split(_fake_conversations())
    assert len(split["train"]) == 5
    assert len(split["val"]) == 2
    assert len(split["test"]) == 3


def test_make_split_partitions_all_conversations_exactly_once():
    convs = _fake_conversations()
    split = make_split(convs)
    all_ids = split["train"] + split["val"] + split["test"]
    assert sorted(all_ids) == sorted(c["sample_id"] for c in convs)
    assert len(all_ids) == len(set(all_ids))


def test_make_split_is_deterministic():
    convs = _fake_conversations()
    assert make_split(convs) == make_split(convs)


def test_make_split_beats_a_bad_assignment_on_balance_score():
    convs = _fake_conversations()
    counts = category_counts_by_conversation(convs)
    overall = {}
    for per_cat in counts.values():
        for cat, n in per_cat.items():
            overall[cat] = overall.get(cat, 0) + n
    total = sum(overall.values())
    overall_fraction = {cat: n / total for cat, n in overall.items()}

    good = make_split(convs)
    good_score = _category_balance_score(good, counts, overall_fraction)

    ids = [c["sample_id"] for c in convs]
    bad = {"train": ids[:5], "val": ids[5:7], "test": ids[7:]}
    bad_score = _category_balance_score(bad, counts, overall_fraction)

    assert good_score <= bad_score
