"""Every committed prompt version (src/amem_gepa/prompts/<component>.v<N>.txt,
CLAUDE.md's prompt-versioning rule) must still be a valid candidate
component -- right placeholders, parseable as a str.format template -- so a
later stage can seed from it with --seed-file without a runtime surprise."""

import re
from pathlib import Path

import pytest

from amem_gepa.repro_injection import COMPONENTS, load_baseline_candidate, validate_candidate

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "src" / "amem_gepa" / "prompts"
VERSIONED = sorted(
    p for p in PROMPTS_DIR.glob("*.v*.txt") if re.fullmatch(r"(.+)\.v\d+\.txt", p.name)
)


@pytest.mark.parametrize("path", VERSIONED, ids=lambda p: p.name)
def test_versioned_prompt_is_a_valid_component(path):
    component = re.fullmatch(r"(.+)\.v\d+\.txt", path.name).group(1)
    assert component in COMPONENTS, f"{path.name} does not name a known component"
    candidate = load_baseline_candidate()
    candidate[component] = path.read_text()
    assert validate_candidate(candidate) == []


def test_at_least_stage_1_winner_is_committed():
    assert (PROMPTS_DIR / "note_construction.v1.txt").exists()
