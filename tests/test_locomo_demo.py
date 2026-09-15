import json

from amem_gepa.datasets.locomo import load_demo_sample


def _fake_raw_conversation(sample_id: str):
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1:00 pm on 1 January, 2024",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I love hiking."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Nice, where?"},
                {"speaker": "Alice", "dia_id": "D1:3", "text": "Yosemite, every summer."},
            ],
            "session_2_date_time": "2:00 pm on 2 January, 2024",
            "session_2": [
                {"speaker": "Bob", "dia_id": "D2:1", "text": "I got a new job."},
                {"speaker": "Alice", "dia_id": "D2:2", "text": "Congrats! Doing what?"},
                {"speaker": "Bob", "dia_id": "D2:3", "text": "Software engineer at a startup."},
            ],
        },
        "qa": [
            {"question": "Where does Alice hike?", "answer": "Yosemite", "category": 1, "evidence": ["D1:3"]},
            {"question": "When does Alice hike?", "answer": "every summer", "category": 2, "evidence": ["D1:3"]},
            {"question": "What is Bob's new job?", "answer": "software engineer", "category": 1, "evidence": ["D2:3"]},
            {"question": "Unanswerable?", "answer": None, "adversarial_answer": "trap", "category": 5, "evidence": ["D2:3"]},
        ],
    }


def _write_fixture(tmp_path, conversation_ids=("conv-a",)):
    raw_path = tmp_path / "locomo10.json"
    raw_path.write_text(json.dumps([_fake_raw_conversation(cid) for cid in conversation_ids]))
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps({"train": [conversation_ids[0]], "val": [], "test": []}))
    return raw_path, manifest_path


def test_demo_sample_truncates_turns(tmp_path):
    raw_path, manifest_path = _write_fixture(tmp_path)

    instances = load_demo_sample(max_turns=3, max_questions=5, manifest_path=manifest_path, raw_path=raw_path)

    assert instances
    assert all(len(inst.turns) == 3 for inst in instances)
    assert [t.dia_id for t in instances[0].turns] == ["D1:1", "D1:2", "D1:3"]


def test_demo_sample_only_includes_answerable_questions(tmp_path):
    raw_path, manifest_path = _write_fixture(tmp_path)

    # Only the first 3 turns (session 1) are kept -- questions whose evidence
    # is in session 2 (D2:*) must be excluded, not asked against missing context.
    instances = load_demo_sample(max_turns=3, max_questions=5, manifest_path=manifest_path, raw_path=raw_path)

    questions = {inst.question for inst in instances}
    assert "Where does Alice hike?" in questions
    assert "When does Alice hike?" in questions
    assert "What is Bob's new job?" not in questions
    assert "Unanswerable?" not in questions


def test_demo_sample_defaults_to_first_train_conversation(tmp_path):
    raw_path, manifest_path = _write_fixture(tmp_path, conversation_ids=("conv-a", "conv-b"))

    instances = load_demo_sample(max_turns=6, max_questions=5, manifest_path=manifest_path, raw_path=raw_path)

    assert all(inst.conversation_id == "conv-a" for inst in instances)


def test_demo_sample_respects_max_questions(tmp_path):
    raw_path, manifest_path = _write_fixture(tmp_path)

    instances = load_demo_sample(max_turns=6, max_questions=2, manifest_path=manifest_path, raw_path=raw_path)

    assert len(instances) == 2


def test_demo_sample_round_robins_categories(tmp_path):
    raw_path, manifest_path = _write_fixture(tmp_path)

    # All 4 questions are answerable with the full 6 turns; category 1 has
    # two candidates (hike location, Bob's job) but round-robin should pull
    # from categories 2 and 5 too before taking a second category-1 one.
    instances = load_demo_sample(max_turns=6, max_questions=3, manifest_path=manifest_path, raw_path=raw_path)

    categories = [inst.category for inst in instances]
    assert categories == sorted(categories), "round-robin should visit categories in ascending order first"
    assert set(categories) == {1, 2, 5}
