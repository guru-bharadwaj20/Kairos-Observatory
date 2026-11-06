import numpy as np
import pytest

from kairos.philosophy.collingridge import LeadTime, first_crossing, lead_time


def growth_paths(n=400, horizon=120, seed=0):
    rng = np.random.default_rng(seed)
    rates = rng.normal(0.05, 0.01, size=(n, 1))
    return np.arange(1, horizon + 1) * rates  # linear in months


def test_first_crossing_simple():
    paths = np.array([[0, 1, 2, 3], [0, 0, 1, 2]], dtype=float)
    assert first_crossing(paths, 1.5, quantile=0.99) == 3
    assert first_crossing(paths, 9.0, quantile=0.99) is None
    assert first_crossing(-paths, -1.5, quantile=0.99, direction="below") == 3


def test_lead_time_inside_interval():
    lt = lead_time(growth_paths(), 2.0, n_boot=300, seed=1)
    assert lt.months is not None
    assert lt.lo is not None
    assert lt.hi is not None
    assert lt.lo <= lt.months <= lt.hi
    # upper quantile of rate ~0.05 + 1.645*0.01 -> crossing near 2/0.0665 ~ 31 months
    assert 27 <= lt.months <= 34
    assert "months (90% CI:" in str(lt)


def test_upper_quantile_is_earlier_than_median():
    paths = growth_paths()
    assert lead_time(paths, 2.0, quantile=0.95, n_boot=50).months < (
        lead_time(paths, 2.0, quantile=0.5, n_boot=50).months
    )


def test_never_crossed_is_censored():
    lt = lead_time(growth_paths(horizon=12), 100.0, n_boot=50)
    assert lt.months is None
    assert lt.hi is None
    assert lt.censored_share == 1.0
    assert str(lt).startswith(">12")


def test_open_upper_end_formatting():
    lt = LeadTime(months=40, lo=30, hi=None, ci=0.9, quantile=0.95, horizon=60, censored_share=0.1)
    assert str(lt) == "40 months (90% CI: 30–>60)"


def test_deterministic_with_seed():
    a = lead_time(growth_paths(), 2.0, n_boot=100, seed=3)
    b = lead_time(growth_paths(), 2.0, n_boot=100, seed=3)
    assert a == b


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros((1, 5)), np.full((3, 3), np.nan)])
def test_bad_paths(bad):
    with pytest.raises(ValueError, match="paths"):
        lead_time(bad, 1.0)
