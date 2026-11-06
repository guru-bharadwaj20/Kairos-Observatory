import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from kairos.philosophy.collingridge import LeadTime
from kairos.philosophy.intervention_cost import InterventionCostCurve
from kairos.philosophy.jonas import load_policy


def policy_curve():
    return InterventionCostCurve.model_validate(load_policy().section("intervention_cost"))


def test_endpoints():
    c = InterventionCostCurve(tau=10, floor=0.2)
    assert c.cost(0) == pytest.approx(1.0)
    assert c.cost(np.inf) == pytest.approx(0.2)
    assert c.cost(10) == pytest.approx(0.2 + 0.8 / np.e)


@given(st.floats(0, 500), st.floats(0, 500))
def test_monotone_decreasing(a, b):
    c = policy_curve()
    lo, hi = sorted((a, b))
    assert c.cost(lo) >= c.cost(hi)


def test_curve_grid():
    months, cost = policy_curve().curve(24)
    assert months.shape == cost.shape == (25,)
    assert np.all(np.diff(cost) < 0)


def test_cost_at_lead_time_interval_order():
    lt = LeadTime(months=31, lo=18, hi=52, ci=0.9, quantile=0.95, horizon=120, censored_share=0)
    out = policy_curve().at(lt)
    assert out.lo < out.cost < out.hi


def test_censored_lead_time():
    lt = LeadTime(
        months=None, lo=None, hi=None, ci=0.9, quantile=0.95, horizon=12, censored_share=1
    )
    out = InterventionCostCurve(tau=5, floor=0.3).at(lt)
    assert out.cost == pytest.approx(0.3)
    assert out.lo == pytest.approx(0.3)


def test_negative_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        policy_curve().cost(-1)
