from __future__ import annotations

import pytest

from kairos.dashboard import components, figures
from kairos.dashboard.snapshot import Alert, LeadTime, load_demo
from kairos.dashboard.theme import LEVELS, level

LEAD = LeadTime(median=31.0, lower=18.0, upper=52.0, interval=0.9, unit="months")


def test_lead_time_text_matches_spec() -> None:
    assert components.lead_time_text(LEAD) == "Lead time: 31 months (90% CI: 18–52)"
    assert "not estimated" in components.lead_time_text(None)


@pytest.mark.parametrize("n", range(5))
def test_banner_shows_level_and_response(n: int) -> None:
    html = components.banner_html(Alert(level=n, lead_time=LEAD))
    spec = level(n)
    assert f"Level {n}: {spec.name}" in html
    assert spec.response.replace("'", "&#x27;") in html
    assert 'role="status"' in html
    assert components.DISCLAIMER in html


def test_snapshot_overrides_and_escapes() -> None:
    alert = Alert(level=1, lead_time=None, name="<Watch>", response="Do X & Y")
    html = components.banner_html(alert)
    assert "&lt;Watch&gt;" in html
    assert "Do X &amp; Y" in html
    assert "<Watch>" not in html


def test_level_bounds() -> None:
    assert [lv.number for lv in LEVELS] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="0-4"):
        level(5)


def test_level_symbols_distinct() -> None:
    # colour is never the only cue
    assert len({lv.symbol for lv in LEVELS}) == 5
    assert len({lv.background for lv in LEVELS}) == 5


def test_scale_marks_current() -> None:
    html = components.level_scale_html(3)
    assert html.count('aria-current="step"') == 1
    assert "3 Alarm</li>" in html


def test_lead_time_figure_has_interval() -> None:
    fig = figures.lead_time_figure(LEAD)
    tr = fig.data[0]
    assert tr.x == (31.0,)
    assert tr.error_x.array == (21.0,)
    assert tr.error_x.arrayminus == (13.0,)
    assert "90%" in fig.layout.title.text


def test_demo_alert() -> None:
    snap = load_demo()
    assert snap.alert.lead_time is not None
