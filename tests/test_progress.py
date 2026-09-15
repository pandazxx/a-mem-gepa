import io

from amem_gepa.progress import ProgressReporter, format_duration


def test_format_duration_variants():
    assert format_duration(5) == "5s"
    assert format_duration(65) == "1m05s"
    assert format_duration(3665) == "1h01m"
    assert format_duration(float("inf")) == "?"


def test_progress_reporter_prints_and_counts():
    out = io.StringIO()
    reporter = ProgressReporter(total=3, label="replay", out=out)

    reporter.tick("conv=conv-x turn=1/3")
    reporter.tick("conv=conv-x turn=2/3")

    lines = out.getvalue().strip().splitlines()
    assert len(lines) == 2
    assert "[replay] 1/3" in lines[0]
    assert "[replay] 2/3" in lines[1]
    assert "turn=2/3" in lines[1]


def test_progress_reporter_resumes_from_done_offset():
    out = io.StringIO()
    reporter = ProgressReporter(total=10, label="answer", done=6, out=out)

    reporter.tick()

    assert "[answer] 7/10" in out.getvalue()
