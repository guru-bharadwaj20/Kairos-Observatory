from __future__ import annotations

import numpy as np
import pytest

from kairos.accel.lppls import (
    fit_lppls,
    lppls_curve,
    lppls_linear,
    lppls_series,
    lppls_sse_cells,
    make_grid,
)


@pytest.fixture(scope="module")
def bubble():
    return lppls_series(120, start="2010-01-01", tc=2020.5, noise=0.005, seed=1)


def test_linear_solve_is_exact_without_noise():
    t = np.linspace(2000, 2010, 100)
    y = lppls_curve(t, 2011.0, 0.4, 7.0, 3.0, -0.8, 0.04, -0.02)
    coef, sse = lppls_linear(t, y, 2011.0, 0.4, 7.0)
    np.testing.assert_allclose(coef, [3.0, -0.8, 0.04, -0.02], atol=1e-8)
    assert sse < 1e-18


def test_centred_cells_match_lstsq(bubble):
    t, y = bubble.t, bubble.log_y
    tc, m, om = make_grid(t[0], t[-1], n_tc=5, n_m=4, n_omega=3)
    sse, coef = lppls_sse_cells(t, y, tc, m, om)
    for k in range(0, len(tc), 7):
        ck, sk = lppls_linear(t, y, tc[k], m[k], om[k])
        assert sse[k] == pytest.approx(sk, rel=1e-6, abs=1e-12)
        np.testing.assert_allclose(coef[k], ck, rtol=1e-5, atol=1e-8)


def test_fit_recovers_known_parameters(bubble):
    fit = fit_lppls(bubble.t, bubble.log_y)
    assert fit.tc == pytest.approx(2020.5, abs=0.2)
    assert fit.m == pytest.approx(0.5, abs=0.1)
    assert fit.omega == pytest.approx(8.0, abs=0.6)
    assert fit.b < 0
    assert fit.rmse < 0.01
    assert fit.n == 120
    assert fit.n_oscillations > 0
    assert fit.damping > 0


def test_grid_rejects_tc_before_window(bubble):
    with pytest.raises(ValueError, match="tc"):
        lppls_linear(bubble.t, bubble.log_y, bubble.t[-2], 0.5, 8.0)


def test_series_validation():
    with pytest.raises(ValueError, match="tc"):
        lppls_series(240, tc=2010.0)
