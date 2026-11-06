from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kairos.philosophy.jonas import load_policy
from kairos.philosophy.probability import DangerRule
from kairos.philosophy.rawls import maximin, worst_case_p_danger
from kairos.types import DialReading

ABOVE = DangerRule(direction="above", threshold=0.0)


def test_maximin_differs_from_expected_value():
    payoffs = {
        "scale": {"benign": 10.0, "bad": -100.0, "mid": 5.0},
        "pause": {"benign": 2.0, "bad": -1.0, "mid": 1.0},
    }
    d = maximin(payoffs, weights={"benign": 0.97, "bad": 0.01, "mid": 0.02})
    assert d.maximin_action == "pause"
    assert d.worst_scenario == "bad"
    assert d.ev_action == "scale"
    assert not d.agree


def test_implausible_scenarios_ignored():
    payoffs = {"a": {"x": 1.0, "y": -50.0}, "b": {"x": 0.0, "y": 0.0}}
    d = maximin(payoffs, weights={"x": 0.99, "y": 0.01}, min_weight=0.05)
    assert d.plausible == ["x"]
    assert d.maximin_action == "a"


def test_mismatched_scenarios_rejected():
    with pytest.raises(ValueError, match="same scenarios"):
        maximin({"a": {"x": 1.0}, "b": {"y": 1.0}})


@given(
    st.dictionaries(
        st.sampled_from("abcd"),
        st.lists(st.floats(-100, 100), min_size=3, max_size=3),
        min_size=1,
    )
)
def test_maximin_value_never_below_any_worst_case(table):
    payoffs = {a: dict(zip("xyz", row, strict=True)) for a, row in table.items()}
    d = maximin(payoffs)
    assert all(d.maximin_value >= min(r.values()) for r in payoffs.values())
    assert d.ev_value >= min(sum(r.values()) / 3 for r in payoffs.values()) - 1e-9


def test_worst_case_p_danger():
    r = DialReading(
        dial="jonas",
        value=0.0,
        level=0,
        ci_lo=0.0,
        ci_hi=0.0,
        as_of=date(2026, 11, 1),
        detail={
            "scenarios": {"slow": [-1, -1, -1, 1], "fast": [1, 1, 1, -1], "rare": [1] * 4},
            "scenario_weights": {"slow": 0.6, "fast": 0.38, "rare": 0.02},
        },
    )
    out = worst_case_p_danger(r, ABOVE, min_weight=load_policy().section("rawls")["min_weight"])
    assert out.worst_scenario == "fast"
    assert out.worst == pytest.approx(0.75)
    assert out.expected == pytest.approx(0.6 * 0.25 + 0.38 * 0.75 + 0.02)
    assert out.worst >= out.expected


def test_worst_case_needs_scenarios():
    r = DialReading(dial="j", value=0, level=0, ci_lo=0, ci_hi=0, as_of=date(2026, 1, 1))
    with pytest.raises(ValueError, match="scenarios"):
        worst_case_p_danger(r, ABOVE)
