import os

import pytest

from amem_gepa.config import load_config


def test_load_config_substitutes_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODEL", "ollama/llama3.2:1b")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("models:\n  amem_llm_model: ${FAKE_MODEL}\n  literal: not-a-var\n")

    config = load_config(config_path)

    assert config["models"]["amem_llm_model"] == "ollama/llama3.2:1b"
    assert config["models"]["literal"] == "not-a-var"


def test_load_config_missing_env_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("models:\n  amem_llm_model: ${DEFINITELY_NOT_SET}\n")

    with pytest.raises(KeyError):
        load_config(config_path)


def test_load_config_real_base_yaml_shape(monkeypatch):
    for var in [
        "AMEM_LLM_MODEL", "AMEM_EMBEDDING_MODEL", "OLLAMA_API_BASE",
        "GEPA_TASK_LM", "GEPA_REFLECTION_LM",
        "PAPER_REPRO_BACKEND", "PAPER_REPRO_MODEL",
    ]:
        monkeypatch.setenv(var, f"fake-{var}")

    config = load_config("configs/base.yaml")

    assert config["evaluation"]["split"] == "test"
    assert config["evaluation"]["retrieval_k"] == 10
    assert config["models"]["amem_llm_model"] == "fake-AMEM_LLM_MODEL"
    assert config["models"]["paper_repro_model"] == "fake-PAPER_REPRO_MODEL"
    assert config["paper_repro"]["retrieve_k"] == 10
