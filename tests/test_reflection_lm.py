"""Tests for the coach (reflection model) backends, docs/decisions/0021:
settings from config, native-Ollama think flag, the Claude request shape
(adaptive thinking + effort, no temperature, no budget_tokens), cost
accounting, refusal handling, and provider dispatch."""

import logging
import types

import pytest

from amem_gepa.reflection_lm import (
    ClaudeReflectionLM,
    OllamaReflectionLM,
    ReflectionSettings,
    build_reflection_lm,
    reflection_stats,
)


def test_settings_from_config_defaults_and_validation():
    s = ReflectionSettings.from_config("ollama/qwen3.5:9b", None)
    assert (s.provider, s.bare_model) == ("ollama", "qwen3.5:9b")
    assert s.thinking is None and s.effort is None and s.max_tokens == 16000
    assert s.max_cost_usd is None and s.fallbacks is True

    s = ReflectionSettings.from_config(
        "anthropic/claude-opus-5-5",
        {"thinking": True, "effort": "xhigh", "max_tokens": 8000, "max_cost_usd": 3, "fallbacks": False},
    )
    assert s.provider == "anthropic" and s.effort == "xhigh" and s.max_cost_usd == 3.0 and not s.fallbacks

    with pytest.raises(ValueError, match="effort"):
        ReflectionSettings.from_config("anthropic/claude-opus-5-5", {"effort": "turbo"})


def test_describe_mentions_the_decisive_settings():
    assert "thinking off" in ReflectionSettings("ollama/qwen3.5:9b", thinking=False).describe()
    assert "model default" in ReflectionSettings("ollama/qwen3.5:9b").describe()
    text = ReflectionSettings("anthropic/claude-opus-5-5", max_cost_usd=2.5).describe()
    assert "effort=high" in text and "$2.50" in text
    assert "LiteLLM" in ReflectionSettings("openai/gpt-4o-mini").describe()


class _FakeOllamaClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "```\nnew prompt {content}\n```", "thinking": "hmm"},
                "prompt_eval_count": 120, "eval_count": 30}


@pytest.mark.parametrize("thinking", [True, False, None])
def test_ollama_coach_passes_think_only_when_configured(thinking):
    client = _FakeOllamaClient()
    lm = OllamaReflectionLM(ReflectionSettings("ollama/qwen3.5:9b", thinking=thinking, max_tokens=512), client=client)
    out = lm("rewrite this")
    call = client.calls[0]
    assert call["model"] == "qwen3.5:9b" and call["messages"] == [{"role": "user", "content": "rewrite this"}]
    assert call["options"] == {"num_predict": 512}
    if thinking is None:
        assert "think" not in call, "null = leave the model's default alone"
    else:
        assert call["think"] is thinking
    assert out.startswith("```"), "content only -- the thinking field is not returned"
    assert lm.calls == 1 and lm.total_tokens_in == 120 and lm.total_tokens_out == 30 and lm.total_cost == 0.0
    assert reflection_stats(lm)["reflection_cost_usd"] == 0.0


def _claude_response(text="```\nnew prompt\n```", stop_reason="end_turn", model="claude-opus-5-5",
                     tokens_in=5000, tokens_out=1500, stop_details=None):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=stop_details,
        model=model,
        usage=types.SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
    )


class _FakeAnthropic:
    def __init__(self, response):
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return response

        self.beta = types.SimpleNamespace(messages=_Messages())


def test_claude_coach_request_shape_and_cost():
    fake = _FakeAnthropic(_claude_response())
    lm = ClaudeReflectionLM(ReflectionSettings("anthropic/claude-opus-5-5", effort="high"), client=fake)
    out = lm("rewrite this")

    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 16000
    assert kw["thinking"] == {"type": "adaptive"}, "adaptive thinking; no budget_tokens on this generation"
    assert kw["output_config"] == {"effort": "high"}
    assert "temperature" not in kw and "budget_tokens" not in str(kw)
    assert kw["betas"] == [ClaudeReflectionLM.FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert kw["messages"] == [{"role": "user", "content": "rewrite this"}]
    assert out == "```\nnew prompt\n```", "text blocks only; thinking blocks dropped"
    # 5000 in @ $4/M + 1500 out @ $20/M
    assert lm.total_cost == pytest.approx(0.02 + 0.03)
    assert reflection_stats(lm) == {
        "reflection_calls": 1, "reflection_tokens_in": 5000, "reflection_tokens_out": 1500,
        "reflection_cost_usd": 0.05,
    }


def test_claude_coach_fallbacks_can_be_disabled_and_thinking_false_means_low_effort(caplog):
    fake = _FakeAnthropic(_claude_response())
    with caplog.at_level(logging.WARNING, logger="amem_gepa.reflection_lm"):
        lm = ClaudeReflectionLM(
            ReflectionSettings("anthropic/claude-opus-5-5", thinking=False, fallbacks=False), client=fake
        )
    assert "effort='low'" in caplog.text
    lm("x")
    kw = fake.calls[0]
    assert kw["output_config"] == {"effort": "low"}
    assert "betas" not in kw and "fallbacks" not in kw


def test_claude_coach_raises_on_refusal_and_logs_a_fallback(caplog):
    refused = _claude_response(
        stop_reason="refusal", stop_details=types.SimpleNamespace(category="cyber", explanation="no")
    )
    lm = ClaudeReflectionLM(ReflectionSettings("anthropic/claude-opus-5-5"), client=_FakeAnthropic(refused))
    with pytest.raises(RuntimeError, match="refused"):
        lm("x")

    served_elsewhere = _claude_response(model="claude-opus-4-8")
    lm = ClaudeReflectionLM(ReflectionSettings("anthropic/claude-opus-5-5"), client=_FakeAnthropic(served_elsewhere))
    with caplog.at_level(logging.WARNING, logger="amem_gepa.reflection_lm"):
        lm("x")
    assert "served by claude-opus-4-8" in caplog.text


def test_claude_coach_warns_once_about_unpriced_models(caplog):
    with caplog.at_level(logging.WARNING, logger="amem_gepa.reflection_lm"):
        lm = ClaudeReflectionLM(ReflectionSettings("anthropic/claude-mystery-9"), client=_FakeAnthropic(_claude_response()))
    assert "no price table entry" in caplog.text
    lm("x")
    assert lm.total_cost == 0.0 and lm.total_tokens_in == 5000


def test_build_reflection_lm_dispatches_on_provider_prefix(monkeypatch):
    fake_ollama = types.ModuleType("ollama")
    fake_ollama.Client = lambda host=None: _FakeOllamaClient()
    monkeypatch.setitem(__import__("sys").modules, "ollama", fake_ollama)
    assert isinstance(build_reflection_lm(ReflectionSettings("ollama/qwen3.5:9b")), OllamaReflectionLM)

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda: _FakeAnthropic(_claude_response())
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic)
    assert isinstance(build_reflection_lm(ReflectionSettings("anthropic/claude-opus-5-5")), ClaudeReflectionLM)

    from gepa.lm import LM

    other = build_reflection_lm(ReflectionSettings("openai/gpt-4o-mini", max_tokens=1234))
    assert isinstance(other, LM) and other.completion_kwargs == {"max_tokens": 1234}
    assert reflection_stats(other)["reflection_cost_usd"] == 0.0
