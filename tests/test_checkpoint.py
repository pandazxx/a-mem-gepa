from amem_gepa.checkpoint import CheckpointStore, ConversationCheckpoint, PredictionStore


def test_checkpoint_store_roundtrip(tmp_path):
    store = CheckpointStore("run-a", results_dir=tmp_path)
    assert store.load("conv-x") is None

    checkpoint = ConversationCheckpoint(
        conversation_id="conv-x",
        turns_processed=3,
        notes=[{"id": "n1", "content": "hi"}],
        completed=False,
    )
    store.save(checkpoint)

    loaded = store.load("conv-x")
    assert loaded == checkpoint


def test_checkpoint_store_overwrites_on_resave(tmp_path):
    store = CheckpointStore("run-a", results_dir=tmp_path)
    store.save(ConversationCheckpoint("conv-x", turns_processed=1, notes=[]))
    store.save(ConversationCheckpoint("conv-x", turns_processed=2, notes=[{"id": "n1"}]))

    loaded = store.load("conv-x")
    assert loaded.turns_processed == 2
    assert loaded.notes == [{"id": "n1"}]


def test_checkpoint_store_separates_conversations(tmp_path):
    store = CheckpointStore("run-a", results_dir=tmp_path)
    store.save(ConversationCheckpoint("conv-a", turns_processed=5))
    store.save(ConversationCheckpoint("conv-b", turns_processed=9))

    assert store.load("conv-a").turns_processed == 5
    assert store.load("conv-b").turns_processed == 9


def test_prediction_store_roundtrip_and_resume(tmp_path):
    store = PredictionStore("run-a", "test", results_dir=tmp_path)
    assert store.get("conv-x", "Q1?") is None
    assert len(store) == 0

    store.append({"conversation_id": "conv-x", "question": "Q1?", "prediction": "answer", "score": 1.0})
    assert len(store) == 1
    assert store.get("conv-x", "Q1?")["prediction"] == "answer"

    # a fresh store pointed at the same path should see prior progress
    resumed = PredictionStore("run-a", "test", results_dir=tmp_path)
    assert len(resumed) == 1
    assert resumed.get("conv-x", "Q1?")["prediction"] == "answer"
    assert resumed.get("conv-x", "Q2?") is None
