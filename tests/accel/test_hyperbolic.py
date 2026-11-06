from __future__ import annotations

import pytest

from kairos.accel.growth import estimate_delta
from kairos.accel.hyperbolic import fit_hyperbolic, fit_hyperbolic_frame
from kairos.accel.synthetic import exponential_series, hyperbolic_series


@pytest.mark.parametrize(("delta", "t_c"), [(0.3, 2022.0), (0.5, 2025.0), (1.0, 2021.0)])
def test_direct_fit_recovers_truth(delta, t_c):
    s = hyperbolic_series(240, delta=delta, t_c=t_c, noise=0.03, seed=5)
    fit = fit_hyperbolic(s.t, s.log_y)
    assert fit.success
    assert not fit.at_bound
    assert fit.delta == pytest.approx(delta, rel=0.05)
    assert fit.t_c == pytest.approx(t_c, abs=0.25 * (t_c - s.t[-1]) + 0.1)
    lo, hi = fit.t_c_ci
    assert lo < fit.t_c < hi


@pytest.mark.parametrize("seed", range(4))
def test_agrees_with_spline_regression(seed):
    s = hyperbolic_series(240, delta=0.5, t_c=2022.0, noise=0.03, seed=seed)
    direct = fit_hyperbolic(s.t, s.log_y)
    spline = estimate_delta(s.t, s.log_y)
    assert direct.delta == pytest.approx(spline.delta, rel=0.08)
    assert direct.t_c == pytest.approx(spline.t_c, abs=0.3)


def test_noise_free_fit_is_exact():
    s = hyperbolic_series(120, delta=0.7, t_c=2012.0, noise=0.0)
    fit = fit_hyperbolic(s.t, s.log_y)
    assert fit.delta == pytest.approx(0.7, rel=1e-4)
    assert fit.t_c == pytest.approx(2012.0, abs=1e-3)
    assert fit.rmse < 1e-6


def test_exponential_pushes_tc_far_away():
    s = exponential_series(240, noise=0.02, seed=1)
    fit = fit_hyperbolic(s.t, s.log_y)
    # exponential is the t_c -> infinity, delta -> 0 limit
    assert fit.t_c - s.t[-1] > 20 or fit.at_bound


def test_frame_wrapper():
    s = hyperbolic_series(180, delta=0.5, t_c=2018.0, seed=2)
    assert fit_hyperbolic_frame(s.frame).delta == pytest.approx(fit_hyperbolic(s.t, s.log_y).delta)
