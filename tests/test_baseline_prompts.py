from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "src" / "amem_gepa" / "prompts" / "baseline"


def test_note_construction_prompt_formats():
    template = (PROMPTS_DIR / "note_construction.txt").read_text()
    formatted = template.format(content="some memory content")
    assert "some memory content" in formatted


def test_evolution_prompt_formats():
    template = (PROMPTS_DIR / "evolution.txt").read_text()
    formatted = template.format(
        content="c",
        context="ctx",
        keywords=["k1", "k2"],
        nearest_neighbors_memories="neighbors",
        neighbor_number=2,
    )
    assert "neighbors" in formatted
    # verbatim-copied JSON example must stay literal braces after .format(),
    # not get consumed as a placeholder
    assert '"should_evolve"' in formatted
