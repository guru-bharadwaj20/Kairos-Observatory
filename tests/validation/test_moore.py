from kairos.validation import run_case
from kairos.validation.moore import load_transistors


def test_snapshot_loads():
    frame = load_transistors()
    assert frame["ds"].dt.year.min() == 1971
    assert frame["y"].is_monotonic_increasing


def test_moore_stays_at_level_zero():
    result = run_case("moore")
    assert result.passed, result.summary
    assert result.summary["max_level"] == 0
    assert result.summary["as_of_dates"] == 37
