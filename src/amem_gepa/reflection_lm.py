"""The reflection ("coach") model behind GEPA's proposals (docs/decisions/0021).

GEPA accepts any callable `(prompt) -> str` as `reflection_lm`. Until 0021
this project passed the raw `GEPA_REFLECTION_LM` string and let gepa route
it through LiteLLM, which gave no control over thinking mode (an Ollama
reasoning model such as qwen3.5:9b then thinks by default, which is what
experiments 004/005 ran with) and no first-class Anthropic configuration.

Three backends, chosen by the model string's provider prefix:

- `ollama/<model>`: the native `ollama` client (the same package the task
  model already uses, docs/decisions/0016), so `think` is a real parameter
  rather than something LiteLLM may or may not forward. `OLLAMA_HOST` is
  honoured like it is for the task model.
- `anthropic/<model>`: the official `anthropic` SDK. Thinking is adaptive
  (always on for the current Claude generation); depth is `effort`. Cost is
  computed from the response's token usage and a small price table so the
  run's experiment doc can state what the coach cost and gepa's
  `max_reflection_cost` can cap it.
- anything else: gepa's own LiteLLM-backed `LM`, unchanged from before.

Every backend exposes `total_cost`, `total_tokens_in`, `total_tokens_out`
and `calls`: gepa uses `total_cost` directly when present, and the run
script writes all four into `summary.json`.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger("amem_gepa.reflection_lm")

# USD per million tokens (input, output), Anthropic first-party API. The
# coach makes ~1 call per GEPA iteration, so a stage costs cents to a few
# dollars; the table exists so the number is reported, not estimated.
ANTHROPIC_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

ANTHROPIC_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


@dataclass
class ReflectionSettings:
    """Everything that decides how the coach is called. `model` is the
    `GEPA_REFLECTION_LM` string; the rest comes from `gepa.reflection` in
    the run config."""

    model: str
    # Ollama: the `think` flag; None = the model's own default (qwen3.5 ->
    # on). Anthropic: thinking is always on for current models; False maps
    # to effort "low" with a warning rather than being rejected.
    thinking: bool | None = None
    # Anthropic only: low | medium | high | xhigh | max. None -> "high".
    effort: str | None = None
    max_tokens: int = 16000
    # Stop the GEPA run when cumulative coach spend exceeds this (USD).
    # Only meaningful for metered backends; Ollama always reports $0.
    max_cost_usd: float | None = None
    # Anthropic only: server-side refusal fallback ("default" routing). On
    # by default per the Claude API guidance; a fallback is logged with the
    # model that actually answered, so the experiment doc can record it.
    fallbacks: bool = True
    # Ollama only: HTTP timeout per coach call, seconds. None = wait forever
    # (the ollama client's own default). A large thinking model that spills
    # to CPU can take an hour per call and look like a hang; a timeout turns
    # that into an error gepa logs and skips.
    request_timeout_s: float | None = None

    @classmethod
    def from_config(cls, model: str, reflection_cfg: dict[str, Any] | None) -> ReflectionSettings:
        cfg = dict(reflection_cfg or {})
        effort = cfg.get("effort")
        if effort is not None and effort not in ANTHROPIC_EFFORT_LEVELS:
            raise ValueError(f"gepa.reflection.effort must be one of {ANTHROPIC_EFFORT_LEVELS}, got {effort!r}")
        thinking = cfg.get("thinking")
        if thinking is not None:
            thinking = bool(thinking)
        return cls(
            model=model,
            thinking=thinking,
            effort=effort,
            max_tokens=int(cfg.get("max_tokens", 16000)),
            max_cost_usd=None if cfg.get("max_cost_usd") is None else float(cfg["max_cost_usd"]),
            fallbacks=bool(cfg.get("fallbacks", True)),
            request_timeout_s=(
                None if cfg.get("request_timeout_s") is None else float(cfg["request_timeout_s"])
            ),
        )

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0] if "/" in self.model else "litellm"

    @property
    def bare_model(self) -> str:
        return self.model.split("/", 1)[1] if "/" in self.model else self.model

    def describe(self) -> str:
        """One line for the run plan."""
        if self.provider == "ollama":
            think = "model default (on for qwen3.5)" if self.thinking is None else ("on" if self.thinking else "off")
            timeout = f"{self.request_timeout_s:.0f}s" if self.request_timeout_s else "none"
            return (
                f"{self.model} via native ollama, thinking {think}, max_tokens={self.max_tokens}, "
                f"timeout {timeout}, $0"
            )
        if self.provider == "anthropic":
            cap = f", cost cap ${self.max_cost_usd:.2f}" if self.max_cost_usd is not None else ", NO cost cap"
            return (
                f"{self.model} via anthropic SDK, adaptive thinking, effort={_anthropic_effort(self)}, "
                f"max_tokens={self.max_tokens}, fallbacks={'on' if self.fallbacks else 'off'}{cap}"
            )
        return f"{self.model} via LiteLLM (gepa.lm.LM), max_tokens={self.max_tokens}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _anthropic_effort(settings: ReflectionSettings) -> str:
    if settings.effort is not None:
        return settings.effort
    # Thinking cannot be switched off on the current Claude generation;
    # "low" effort is the supported way to ask for the cheapest reasoning.
    return "low" if settings.thinking is False else "high"


class _Tracked:
    """Cost/usage counters every backend exposes. gepa reads `total_cost`
    (and skips its own estimator when the attribute exists)."""

    def __init__(self) -> None:
        self.total_cost: float = 0.0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.calls: int = 0

    def stats(self) -> dict[str, Any]:
        return {
            "reflection_calls": self.calls,
            "reflection_tokens_in": self.total_tokens_in,
            "reflection_tokens_out": self.total_tokens_out,
            "reflection_cost_usd": round(self.total_cost, 4),
        }


class OllamaReflectionLM(_Tracked):
    """Coach on a local Ollama model, with an explicit `think` flag."""

    def __init__(self, settings: ReflectionSettings, client: Any | None = None):
        super().__init__()
        self.settings = settings
        self.model = settings.bare_model
        if client is None:
            import ollama

            # Same endpoint resolution as the task model: the native client
            # reads OLLAMA_HOST (docs/decisions/0013's dual-variable note).
            client = ollama.Client(
                host=os.environ.get("OLLAMA_HOST") or None, timeout=settings.request_timeout_s
            )
        self._client = client

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {"num_predict": self.settings.max_tokens},
        }
        if self.settings.thinking is not None:
            kwargs["think"] = self.settings.thinking
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        # The coach is the one call in a GEPA iteration that the build/QA
        # progress lines do not cover; without these two lines a slow coach
        # (first load of a large model, CPU spill-over) looks like a hang.
        logger.info(
            "[coach] call %d -> %s (think=%s, prompt %d chars, timeout %s) ...",
            self.calls + 1,
            self.model,
            self.settings.thinking,
            prompt_chars,
            f"{self.settings.request_timeout_s:.0f}s" if self.settings.request_timeout_s else "none",
        )
        started = time.monotonic()
        response = self._client.chat(**kwargs)
        elapsed = time.monotonic() - started
        message = response.get("message") or {}
        content = message.get("content") or ""
        self.calls += 1
        tokens_in = int(response.get("prompt_eval_count") or 0)
        tokens_out = int(response.get("eval_count") or 0)
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        logger.info(
            "[coach] call %d done in %.0fs: prompt_eval=%d eval=%d (reply %d chars%s)",
            self.calls,
            elapsed,
            tokens_in,
            tokens_out,
            len(content),
            ", thinking returned separately" if message.get("thinking") else "",
        )
        if self.calls == 1:
            self._log_placement()
        return content

    def _log_placement(self) -> None:
        """After the first call, say where Ollama put the coach model. A
        model larger than free VRAM is split to CPU and runs many times
        slower -- the single most likely reason a coach call takes an hour."""
        try:
            running = getattr(self._client.ps(), "models", None) or []
        except Exception as e:  # noqa: BLE001 -- diagnostics must never fail the run
            logger.info("[coach] could not query `ollama ps`: %s", e)
            return
        for entry in running:
            name = getattr(entry, "model", None) or getattr(entry, "name", "?")
            size = int(getattr(entry, "size", 0) or 0)
            size_vram = int(getattr(entry, "size_vram", 0) or 0)
            if not size:
                continue
            gpu_share = size_vram / size
            line = "[coach] ollama ps: %s %.1f GB, %.0f%% in VRAM"
            if gpu_share < 0.99:
                logger.warning(
                    line + " -- partly on CPU; expect calls many times slower than on GPU",
                    name,
                    size / 1e9,
                    gpu_share * 100,
                )
            else:
                logger.info(line, name, size / 1e9, gpu_share * 100)

    def __repr__(self) -> str:
        return f"OllamaReflectionLM({self.model!r}, think={self.settings.thinking})"


class ClaudeReflectionLM(_Tracked):
    """Coach on Claude through the official `anthropic` SDK."""

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, settings: ReflectionSettings, client: Any | None = None):
        super().__init__()
        self.settings = settings
        self.model = settings.bare_model or DEFAULT_ANTHROPIC_MODEL
        self.effort = _anthropic_effort(settings)
        if settings.thinking is False and settings.effort is None:
            logger.warning(
                "gepa.reflection.thinking=false: Claude cannot switch thinking off; using effort='low' instead"
            )
        if self._price() is None:
            logger.warning(
                "no price table entry for %s -- reflection_cost_usd will read 0; add it to "
                "ANTHROPIC_PRICING_USD_PER_MTOK",
                self.model,
            )
        if client is None:
            import anthropic

            # Credentials come from the environment (ANTHROPIC_API_KEY, or an
            # `ant auth login` profile); never from config files.
            client = anthropic.Anthropic()
        self._client = client

    def _price(self) -> tuple[float, float] | None:
        for key, price in ANTHROPIC_PRICING_USD_PER_MTOK.items():
            if self.model == key or self.model.startswith(key + "-"):
                return price
        return None

    def request_kwargs(self, prompt: str | list[dict[str, Any]]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.max_tokens,
            # Adaptive thinking is the only on-mode for the current Claude
            # generation; depth is controlled by effort, not a token budget.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            "messages": messages,
        }
        if self.settings.fallbacks:
            kwargs["betas"] = [self.FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        kwargs = self.request_kwargs(prompt)
        logger.info("[coach] call %d -> %s (effort=%s) ...", self.calls + 1, self.model, self.effort)
        started = time.monotonic()
        response = self._client.beta.messages.create(**kwargs)
        elapsed = time.monotonic() - started
        self.calls += 1
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        price = self._price()
        if price is not None:
            self.total_cost += (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000
        logger.info(
            "[coach] call %d done in %.0fs: in=%d out=%d, run total $%.3f",
            self.calls,
            elapsed,
            tokens_in,
            tokens_out,
            self.total_cost,
        )
        served_by = getattr(response, "model", self.model)
        if served_by and served_by != self.model:
            logger.warning(
                "reflection call %d was served by %s after a refusal fallback (requested %s)",
                self.calls,
                served_by,
                self.model,
            )
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(
                f"reflection model {served_by} refused the request"
                + (f" ({getattr(details, 'category', None)}: {getattr(details, 'explanation', None)})" if details else "")
            )
        if getattr(response, "stop_reason", None) == "max_tokens":
            logger.warning(
                "reflection response truncated at max_tokens=%d; raise gepa.reflection.max_tokens",
                self.settings.max_tokens,
            )
        return "".join(
            getattr(block, "text", "") for block in response.content if getattr(block, "type", None) == "text"
        )

    def __repr__(self) -> str:
        return f"ClaudeReflectionLM({self.model!r}, effort={self.effort!r})"


def build_reflection_lm(settings: ReflectionSettings):
    """The callable to hand gepa as `reflection_lm`, chosen by provider
    prefix. Also returns a stats() provider via the `_Tracked` interface
    for the two native backends; the LiteLLM fallback exposes gepa's own
    total_cost/total_tokens_* instead."""
    if settings.provider == "ollama":
        return OllamaReflectionLM(settings)
    if settings.provider == "anthropic":
        return ClaudeReflectionLM(settings)
    from gepa.lm import LM

    return LM(settings.model, max_tokens=settings.max_tokens)


def reflection_stats(lm: Any) -> dict[str, Any]:
    """Uniform stats dict for summary.json whatever backend `lm` is."""
    if isinstance(lm, _Tracked):
        return lm.stats()
    return {
        "reflection_calls": None,
        "reflection_tokens_in": getattr(lm, "total_tokens_in", None),
        "reflection_tokens_out": getattr(lm, "total_tokens_out", None),
        "reflection_cost_usd": getattr(lm, "total_cost", None),
    }
