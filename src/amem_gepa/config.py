"""Loads a configs/*.yaml run config, substituting ${ENV_VAR} placeholders
(see configs/base.yaml) from the process environment -- .env is expected to
already be loaded (each script calls dotenv.load_dotenv() first) before this
runs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name not in os.environ:
                raise KeyError(f"Config references ${{{var_name}}}, but it's not set in the environment")
            return os.environ[var_name]

        return _ENV_VAR_PATTERN.sub(replace, value)
    return value


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    return _substitute(raw)
