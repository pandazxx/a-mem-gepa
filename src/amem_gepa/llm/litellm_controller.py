"""Drop-in replacement for agentic_memory.llm_controller.LLMController.

Upstream A-MEM's LLMController (external/a-mem/agentic_memory/llm_controller.py)
only accepts backend="openai" or "ollama" -- there's no way to pass an
Anthropic or NVIDIA NIM model string through it. Rather than fork that file,
AgenticMemorySystem instances get their `.llm_controller` attribute swapped
for one of these post-construction (see amem_adapter.py); both expose the
same `get_completion(prompt, response_format, temperature)` shape the rest
of memory_system.py calls.
"""

from __future__ import annotations

import json
from typing import Optional

import litellm


class LiteLLMBackend:
    def __init__(self, model: str, api_base: Optional[str] = None):
        self.model = model
        self.api_base = api_base

    def get_completion(
        self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.7
    ) -> str:
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            temperature=temperature,
            api_base=self.api_base,
        )
        return response.choices[0].message.content


class LiteLLMController:
    """Matches agentic_memory.llm_controller.LLMController's public shape."""

    def __init__(self, model: str, api_base: Optional[str] = None):
        self.llm = LiteLLMBackend(model, api_base)

    def get_completion(
        self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.7
    ) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)


def parse_json_response(raw: str) -> dict:
    """A-MEM's own analyze_content/process_memory do a bare json.loads(response)
    with no fence-stripping; some providers wrap JSON in ```json fences even
    when asked not to. Centralize that tolerance here instead of upstream."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return json.loads(text)
