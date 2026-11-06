from __future__ import annotations

from dataclasses import replace

import pytest

from kairos.dashboard import components, figures
from kairos.dashboard.snapshot import DIAL_ORDER, load_demo
from kairos.dashboard.theme import DIALS, STATE_STYLE


@pytest.mark.parametrize("name", DIAL_ORDER)
def test_card_states_in_words(name: str) -> None:
    d = load_demo().dial(name)  # type: ignore[arg-type]
    for state in ("calm", "danger"):
        html = components.dial_card_html(replace(d, state=state))
        assert STATE_STYLE[state][2] in html
        assert DIALS[name].title in html
        assert "P(danger)" in html


def test_card_escapes_symbol() -> None:
    d = replace(load_demo().dial("recursion"), symbol="<k>")
    assert "&lt;k&gt;" in components.dial_card_html(d)


def test_sparkline_uses_history_and_state_colour() -> None:
    d = load_demo().dial("jonas_gap")
    fig = figures.sparkline(d)
    assert list(fig.data[0].y) == list(d.history.value)
    assert fig.data[0].line.color == STATE_STYLE[d.state][0]


def test_state_styles_differ_beyond_colour() -> None:
    calm, danger = STATE_STYLE["calm"], STATE_STYLE["danger"]
    assert calm[1] != danger[1]
    assert calm[2] != danger[2]
