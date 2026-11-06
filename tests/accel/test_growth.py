from __future__ import annotations

import numpy as np
import pytest

from kairos.accel.growth import (
    delta_from_frame,
    edge_mask,
    estimate_delta,
    ols_delta,
    select_lambda,
    spline_smoother,
)
from kairos.accel.synthetic import exponential_series, hyperbolic_series, logistic_series


def test_smoother_operators_reproduce_lines():
    t = np.linspace(2000, 2020, 120)
    sm = spline_smoother(t, 1.0)
    line = 3.0 + 0.5 * (t - 2000)
    np.testing.assert_allclose(sm.smooth(line), line, atol=1e-8)
    np.testing.assert_allclose(sm.derivative(line), 0.5, atol=1e-8)
    assert 2.0 < sm.edf < 120.0


def test_smoother_batches_rows():
    t = np.linspace(0, 10, 50)
    sm = spline_smoother(t, 0.1)
    z = np.random.default_rng(0).normal(size=(4, 50))
    np.testing.assert_allclose(sm.derivative(z)[2], sm.derivative(z[2]))


def test_gcv_prefers_more_smoothing_for_noisier_data():
    s_clean = hyperbolic_series(240, noise=0.005, seed=1)
    s_noisy = hyperbolic_series(240, noise=0.1, seed=1)
    assert select_lambda(s_noisy.t, s_noisy.log_y) > select_lambda(s_clean.t, s_clean.log_y)


@pytest.mark.parametrize("delta", [0.3, 0.5, 1.0])
def test_recovers_hyperbolic_delta_and_tc(delta):
    s = hyperbolic_series(240, delta=delta, t_c=2022.0, noise=0.03, seed=1)
    fit = estimate_delta(s.t, s.log_y)
    assert fit.delta == pytest.approx(delta, rel=0.05)
    assert fit.t_c == pytest.approx(2022.0, abs=0.3)
    assert fit.t_c_ci[0] < fit.t_c < fit.t_c_ci[1]
    # D5 closed form: t_c - t = 1/(delta g)
    assert fit.t_remaining == pytest.approx(fit.t_c - fit.t_last)


def test_exponential_gives_delta_near_zero():
    s = exponential_series(240, noise=0.05, seed=2)
    fit = estimate_delta(s.t, s.log_y)
    assert abs(fit.delta) < 0.05
    if fit.delta <= 0:
        assert np.isnan(fit.t_c)


def test_logistic_gives_negative_delta_and_no_tc():
    s = logistic_series(240, seed=3)
    fit = estimate_delta(s.t, s.log_y)
    assert fit.delta < 0
    assert np.isnan(fit.t_remaining)
    assert np.isnan(fit.t_c_ci[0])


def test_ols_delta_batched_matches_scalar():
    rng = np.random.default_rng(0)
    lx = rng.normal(size=(3, 30)).cumsum(axis=1)
    g = np.exp(0.2 + 0.7 * lx + 0.01 * rng.normal(size=(3, 30)))
    w = edge_mask(30, 2)
    a, d, n = ols_delta(lx, g, w)
    for i in range(3):
        ai, di, _ = ols_delta(lx[i], g[i], w)
        assert d[i] == pytest.approx(float(di))
        assert a[i] == pytest.approx(float(ai))
    np.testing.assert_allclose(d, 0.7, atol=0.01)
    assert (n == 26).all()


def test_ols_delta_ignores_non_positive_growth():
    lx = np.linspace(0, 1, 10)
    g = np.exp(lx)
    g[3] = -1.0
    _, d, n = ols_delta(lx, g, np.ones(10))
    assert float(d) == pytest.approx(1.0)
    assert float(n) == 9


def test_delta_from_frame_uses_ds_and_y():
    s = hyperbolic_series(180, delta=0.5, t_c=2018.0, seed=4)
    fit = delta_from_frame(s.frame)
    direct = estimate_delta(s.t, s.log_y)
    assert fit.delta == pytest.approx(direct.delta)


def test_input_validation():
    t = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="increasing"):
        estimate_delta(t[::-1], t)
    with pytest.raises(ValueError, match="finite"):
        estimate_delta(t, np.full(20, np.nan))
    with pytest.raises(ValueError, match="positive growth"):
        estimate_delta(t, -5 * t, lam=1.0)
