from datetime import date

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from kairos.philosophy.jonas import load_policy
from kairos.philosophy.probability import (
    DangerRule,
    MissingDistributionError,
    danger_rules,
    p_danger,
    p_danger_from_samples,
)
from kairos.types import DialReading

ABOVE = DangerRule(direction="above", threshold=0.0)
BELOW = DangerRule(direction="below", threshold=0.0)


def reading(value=0.1, lo=-0.1, hi=0.3, **detail):
    return DialReading(
        dial="acceleration",
        value=value,
        level=0,
        ci_lo=lo,
        ci_hi=hi,
        as_of=date(2026, 11, 1),
        detail=detail,
    )


def test_share_of_samples():
    r = reading(samples=[-1.0, 1.0, 2.0, 3.0])
    out = p_danger(r, ABOVE)
    assert out.p == pytest.approx(0.75)
    assert out.n == 4
    assert out.method == "samples"
    assert out.lo < 0.75 < out.hi


def test_below_direction_and_nan_dropped():
    out = p_danger_from_samples("vinge", [-1.0, float("nan"), 1.0], BELOW)
    assert out.p == pytest.approx(0.5)
    assert out.n == 2


def test_point_estimate_refused():
    with pytest.raises(MissingDistributionError, match="point estimate"):
        p_danger(reading(), ABOVE)


def test_normal_fallback_is_labelled():
    out = p_danger(reading(value=0.0, lo=-1, hi=1), ABOVE, allow_normal_fallback=True)
    assert out.method == "normal-ci"
    assert out.p == pytest.approx(0.5)


def test_fallback_agrees_with_samples():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.5, 1.0, 200_000)
    lo, hi = np.quantile(draws, [0.05, 0.95])
    by_ci = p_danger(reading(0.5, lo, hi), ABOVE, allow_normal_fallback=True).p
    by_samples = p_danger(reading(0.5, lo, hi, samples=draws.tolist()), ABOVE).p
    assert by_ci == pytest.approx(by_samples, abs=0.01)


@given(st.lists(st.floats(-10, 10), min_size=1, max_size=50))
def test_directions_complement(xs):
    xs = [x for x in xs if x != 0.0] or [1.0]
    a = p_danger_from_samples("d", xs, ABOVE).p
    b = p_danger_from_samples("d", xs, BELOW).p
    assert a + b == pytest.approx(1.0)


def test_policy_has_rules_for_all_dials():
    rules = danger_rules(load_policy().section("danger"))
    assert set(rules) == {"acceleration", "recursion", "vinge", "jonas"}
    assert rules["vinge"].direction == "below"
