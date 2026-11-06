from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.accel.synthetic import (
    ar1_noise,
    decimal_years,
    exponential_series,
    hyperbolic_series,
    logistic_series,
    month_starts,
)
from kairos.types import validate_series_frame


def test_decimal_years_month_starts():
    ds = month_starts("2020-01-15", 13)
    t = decimal_years(ds)
    assert t[0] == 2020.0
    assert t[-1] == 2021.0
    assert np.isclose(t[6], 2020.0 + 182 / 366)


@pytest.mark.parametrize("gen", [exponential_series, hyperbolic_series, logistic_series])
def test_frames_follow_convention(gen):
    s = gen(120, seed=3)
    validate_series_frame(s.frame)
    assert len(s.frame) == 120
    assert s.frame["ds"].is_monotonic_increasing
    assert (s.frame["source_group"] == "synthetic").all()


def test_same_seed_same_series():
    a = hyperbolic_series(seed=7).frame
    b = hyperbolic_series(seed=7).frame
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["y"], hyperbolic_series(seed=8).frame["y"])


def test_hyperbolic_growth_law_exact():
    s = hyperbolic_series(240, delta=0.5, t_c=2025.0, noise=0.0)
    g = np.gradient(s.log_x_true, s.t)
    slope = np.polyfit(s.log_x_true[5:-5], np.log(g[5:-5]), 1)[0]
    assert slope == pytest.approx(0.5, abs=5e-3)
    np.testing.assert_allclose(g[5:-5], 1 / (0.5 * (2025.0 - s.t[5:-5])), rtol=2e-2)


def test_exponential_constant_growth_rate():
    s = exponential_series(120, rate=0.3, noise=0.0)
    np.testing.assert_allclose(np.diff(s.log_x_true) / np.diff(s.t), 0.3)
    assert s.params["delta"] == 0.0


def test_logistic_growth_rate_falls_with_level():
    s = logistic_series(240, noise=0.0)
    g = np.gradient(s.log_x_true, s.t)
    assert np.polyfit(s.log_x_true[120:], np.log(g[120:]), 1)[0] < 0


def test_hyperbolic_rejects_points_past_singularity():
    with pytest.raises(ValueError, match="before t_c"):
        hyperbolic_series(240, t_c=2010.0)
    with pytest.raises(ValueError, match="delta"):
        hyperbolic_series(delta=0.0)


def test_ar1_noise_moments():
    e = ar1_noise(200_000, 0.1, 0.6, np.random.default_rng(0))
    assert e.std() == pytest.approx(0.1, rel=0.02)
    assert np.corrcoef(e[:-1], e[1:])[0, 1] == pytest.approx(0.6, abs=0.01)
