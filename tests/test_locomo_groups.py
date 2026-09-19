"""Tests for the GEPA train-group / val-subset builders
(datasets/locomo.py, docs/decisions/0015)."""

from amem_gepa.datasets.locomo import (
    LoCoMoInstance,
    LoCoMoTurn,
    build_question_groups,
    build_val_subset,
)


def _turn(i):
    return LoCoMoTurn(session=1, dia_id=f"D1:{i}", speaker="A", text=f"turn {i}", date_time="t")


def _instances(conv_id, per_category_counts):
    turns = [_turn(i) for i in range(3)]
    out = []
    for category, n in per_category_counts.items():
        for i in range(n):
            out.append(
                LoCoMoInstance(
                    conversation_id=conv_id,
                    turns=turns,
                    question=f"{conv_id}-cat{category}-q{i}",
                    gold_answer=f"a{i}",
                    category=category,
                )
            )
    return out


def test_groups_are_single_conversation_and_respect_quota():
    instances = _instances("conv-a", {1: 2, 2: 7, 4: 7, 5: 7}) + _instances("conv-b", {4: 3})
    groups = build_question_groups(instances, per_category=3)

    for g in groups:
        assert len({q.conversation_id for q in g.questions}) == 1
        assert g.conversation_id == g.questions[0].conversation_id
        for count in g.category_counts.values():
            assert count <= 3


def test_groups_consume_every_question_exactly_once():
    instances = _instances("conv-a", {1: 2, 2: 7, 4: 7, 5: 7}) + _instances("conv-b", {4: 3})
    groups = build_question_groups(instances, per_category=3)

    seen = [q.question for g in groups for q in g.questions]
    assert sorted(seen) == sorted(i.question for i in instances)
    assert len(seen) == len(set(seen))


def test_first_group_is_stratified_and_tail_degrades_gracefully():
    # multi_hop (1) is scarce: 2 questions vs temporal's 7 -- the first
    # group should still carry both multi_hop questions, and later groups
    # just lack the exhausted category (docs/decisions/0015).
    instances = _instances("conv-a", {1: 2, 2: 7})
    groups = build_question_groups(instances, per_category=3)

    assert groups[0].category_counts == {1: 2, 2: 3}
    assert groups[1].category_counts == {2: 3}
    assert groups[2].category_counts == {2: 1}


def test_groups_are_deterministic():
    instances = _instances("conv-a", {1: 2, 2: 7, 4: 7})
    a = build_question_groups(instances, per_category=3)
    b = build_question_groups(instances, per_category=3)
    assert [[q.question for q in g.questions] for g in a] == [[q.question for q in g.questions] for g in b]


def test_val_subset_caps_per_category_per_conversation():
    instances = _instances("conv-a", {2: 40, 4: 10}) + _instances("conv-b", {2: 5})
    subset = build_val_subset(instances, per_category_per_conversation=15)

    by_key = {}
    for inst in subset:
        by_key.setdefault((inst.conversation_id, inst.category), []).append(inst)
    assert len(by_key[("conv-a", 2)]) == 15  # capped, spread by striding
    assert len(by_key[("conv-a", 4)]) == 10  # fewer than the cap: take all
    assert len(by_key[("conv-b", 2)]) == 5

    # Striding spreads across the list rather than taking a prefix: with
    # 40 items and n=15 the last picked index is int(14 * 40/15) = 37,
    # well past the first-15 prefix.
    picked_indices = {int(i.question.rsplit("q", 1)[1]) for i in by_key[("conv-a", 2)]}
    assert max(picked_indices) > 30
    assert picked_indices != set(range(15))


def test_val_subset_is_deterministic_and_unique():
    instances = _instances("conv-a", {2: 40, 4: 10})
    a = build_val_subset(instances, per_category_per_conversation=15)
    b = build_val_subset(instances, per_category_per_conversation=15)
    assert [i.question for i in a] == [i.question for i in b]
    assert len({i.question for i in a}) == len(a)
