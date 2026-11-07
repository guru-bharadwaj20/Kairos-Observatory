from __future__ import annotations

from dataclasses import replace

import pytest

from kairos.dashboard import methodology
from kairos.dashboard.snapshot import load_demo
from kairos.dashboard.theme import LEVELS


def test_alarm_threshold() -> None:
    assert methodology.alarm_threshold(9.0) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="positive"):
        methodology.alarm_threshold(0.0)


def test_page_uses_published_lambda() -> None:
    snap = load_demo()
    snap = replace(snap, parameters=replace(snap.parameters, lam=4.0))
    md = methodology.methodology_markdown(snap)
    assert "λ = 4" in md
    assert "20.0%" in md


def test_page_covers_all_notes_and_levels() -> None:
    md = methodology.methodology_markdown(load_demo())
    for n in range(1, 13):
        assert f"| D{n} |" in md
    for lv in LEVELS:
        assert lv.response in md


def test_tables_are_contiguous() -> None:
    md = methodology.methodology_markdown(load_demo())
    assert "|\n\n|" not in md
