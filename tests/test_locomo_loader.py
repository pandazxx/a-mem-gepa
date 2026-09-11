import json

from amem_gepa.datasets.locomo import _turns_for_conversation, load_split


def _fake_raw_conversation(sample_id: str):
    # Mirrors the real locomo10.json shape: "session_N_date_time" keys exist
    # for sessions beyond what's actually generated (session_N missing), and
    # "session_summary"/"observation" live on the outer record, not inside
    # "conversation" -- a naive `k.split("_")[1].isdigit()` filter matches
    # "session_1_date_time" too (split()[1] == "1"), which is the bug this
    # fixture is shaped to catch.
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1:00 pm on 1 January, 2024",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "Hi Bob!"},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Hi Alice!"},
            ],
            "session_2_date_time": "2:00 pm on 2 January, 2024",
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "How are you?"},
            ],
            # a date placeholder with no matching turn list, like the real data
            "session_3_date_time": "3:00 pm on 3 January, 2024",
        },
        "session_summary": {"session_1_summary": "they greeted each other"},
        "qa": [
            {"question": "Who said hi first?", "answer": "Alice", "category": 1},
            {"question": "Unanswerable?", "answer": None, "adversarial_answer": "trap", "category": 5},
        ],
    }


def test_turns_for_conversation_skips_date_time_and_missing_sessions():
    conv = _fake_raw_conversation("conv-x")["conversation"]
    turns = _turns_for_conversation(conv)
    assert [t.dia_id for t in turns] == ["D1:1", "D1:2", "D2:1"]
    assert turns[0].date_time == "1:00 pm on 1 January, 2024"
    assert turns[2].session == 2


def test_load_split_end_to_end(tmp_path):
    raw_path = tmp_path / "locomo10.json"
    raw_path.write_text(json.dumps([_fake_raw_conversation("conv-x")]))
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps({"train": ["conv-x"], "val": [], "test": []}))

    instances = load_split("train", manifest_path=manifest_path, raw_path=raw_path)

    assert len(instances) == 2
    assert instances[0].conversation_id == "conv-x"
    assert len(instances[0].turns) == 3
    adversarial = [i for i in instances if i.is_adversarial]
    assert len(adversarial) == 1
    assert adversarial[0].gold_answer is None
    assert adversarial[0].adversarial_answer == "trap"
