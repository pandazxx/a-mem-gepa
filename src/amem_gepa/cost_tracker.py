"""Token/cost accounting for a run, so every docs/experiments/NNN-*.md can
report a real $/token figure (CLAUDE.md's cost-awareness section) instead of
an afterthought guess.

Not implemented yet -- wire this into LiteLLMBackend.get_completion
(llm/litellm_controller.py) once a run actually needs the numbers, using
litellm's built-in response usage/cost callbacks rather than re-deriving
token counts by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostTracker:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: list[dict] = field(default_factory=list)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost_usd
        self.calls.append(
            {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
            }
        )

    def summary(self) -> dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "num_calls": len(self.calls),
        }
