import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

from kairos.vinge.trend import mann_kendall_s, shrinkage_by_series, shrinkage_test


def monthly(values, start="2015-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


def test_s_matches_kendall_tau():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    s, var = mann_kendall_s(x)
    tau = kendalltau(np.arange(40), x).statistic
    assert s / (40 * 39 / 2) == pytest.approx(tau)
    assert var == pytest.approx(40 * 39 * 85 / 18)


def test_tie_correction_reduces_variance():
    _, var_ties = mann_kendall_s([1, 1, 1, 2, 2, 3])
    _, var_none = mann_kendall_s([1, 2, 3, 4, 5, 6])
    assert var_ties < var_none


def test_detects_shrinking_horizon_with_slope_in_months_per_year():
    rng = np.random.default_rng(1)
    h = 24 - 0.25 * np.arange(60) + rng.normal(0, 1.0, 60)  # -3 months per year
    res = shrinkage_test(monthly(h))
    assert res.shrinking
    assert res.p_value < 0.01
    assert res.slope == pytest.approx(-3.0, abs=0.5)
    assert res.slope_lo <= res.slope <= res.slope_hi


def test_flat_noise_not_significant_and_constant_is_safe():
    rng = np.random.default_rng(2)
    assert not shrinkage_test(monthly(rng.normal(12, 1, 60))).shrinking
    res = shrinkage_test(monthly([6.0] * 10))
    assert res.p_value == pytest.approx(0.5)
    assert not res.shrinking


def test_growing_horizon_is_not_shrinkage():
    res = shrinkage_test(monthly(np.arange(30, dtype=float)))
    assert not res.shrinking
    assert res.p_value > 0.99


def test_hamed_rao_is_more_conservative_on_autocorrelated_noise():
    rng = np.random.default_rng(5)
    # Random walk: strong autocorrelation, no deterministic trend.
    walk = np.cumsum(rng.normal(size=80))
    naive = shrinkage_test(monthly(walk), correction="none")
    robust = shrinkage_test(monthly(walk), correction="hamed_rao")
    assert abs(robust.z) <= abs(naive.z)


def test_by_series_and_errors():
    df = pd.concat(
        [
            pd.DataFrame(
                {"series_id": "a", "as_of": monthly(range(10)).index, "h_star": range(10, 0, -1)}
            ),
            pd.DataFrame({"series_id": "b", "as_of": monthly(range(3)).index, "h_star": [1, 2, 3]}),
        ]
    )
    out = shrinkage_by_series(df)
    assert list(out["series_id"]) == ["a"]
    assert bool(out["shrinking"].iloc[0])
    with pytest.raises(ValueError, match="at least 4"):
        shrinkage_test(monthly([1.0, 2.0]))
    with pytest.raises(ValueError, match="alpha"):
        shrinkage_test(monthly([1.0] * 5), alpha=1.5)
