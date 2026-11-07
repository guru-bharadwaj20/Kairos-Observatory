from __future__ import annotations

import numpy as np
import pytest

from kairos.accel.lppls import lppls_linear, lppls_series, lppls_sse_cells
from kairos.accel.lppls_gpu import lppls_grid_search, normalised_grid

GRID = {"n_tc": 8, "n_m": 6, "n_omega": 6}


@pytest.fixture(scope="module")
def bubble():
    return lppls_series(160, start="2008-01-01", tc=2021.6, noise=0.01, seed=3)


def _windows(n):
    return [(s, n) for s in range(0, 80, 20)] + [(10, n - 12)]


def _reference(t, y, windows):
    frac, m, om = normalised_grid(**GRID)
    out = []
    for a, b in windows:
        tw = t[a:b]
        tc = tw[-1] + frac * (tw[-1] - tw[0])
        out.append(lppls_sse_cells(tw, y[a:b], tc, m, om)[0])
    return np.array(out)


def _check(res, ref, t, y, tol):
    tss = np.array([((y[a:b] - y[a:b].mean()) ** 2).sum() for a, b in res.windows])
    np.testing.assert_allclose(res.sse_grid, ref, atol=tol * tss.max(), rtol=0)
    for w, (a, b) in enumerate(res.windows):
        _, sse64 = lppls_linear(t[a:b], y[a:b], res.tc[w, 0], res.m[w, 0], res.omega[w, 0])
        assert sse64 <= ref[w].min() + tol * tss[w]


def test_torch_cpu_float64_equals_reference(bubble):
    t, y = bubble.t, bubble.log_y
    wins = _windows(len(t))
    res = lppls_grid_search(t, y, wins, device="cpu", return_grid=True, top_k=3, **GRID)
    assert res.sse.shape == (len(wins), 3)
    assert (np.diff(res.sse, axis=1) >= 0).all()
    _check(res, _reference(t, y, wins), t, y, 1e-9)


def test_best_fits_recover_bubble(bubble):
    t, y = bubble.t, bubble.log_y
    res = lppls_grid_search(t, y, [(0, len(t))], device="cpu", n_tc=24, n_m=12, n_omega=12)
    (fit,) = res.best_fits(t, y)
    assert fit.tc == pytest.approx(2021.6, abs=0.5)
    assert fit.b < 0


def test_validation(bubble):
    with pytest.raises(ValueError, match="windows"):
        lppls_grid_search(bubble.t, bubble.log_y, [(0, 5)], device="cpu")
    with pytest.raises(ValueError, match="window"):
        lppls_grid_search(bubble.t, bubble.log_y, [], device="cpu")


@pytest.mark.gpu
def test_gpu_float32_matches_float64_reference(bubble):
    t, y = bubble.t, bubble.log_y
    wins = _windows(len(t))
    res = lppls_grid_search(t, y, wins, device="cuda", return_grid=True, **GRID)
    assert res.device.startswith("cuda")
    _check(res, _reference(t, y, wins), t, y, 1e-4)


@pytest.mark.gpu
def test_gpu_large_grid_chunks(bubble):
    t, y = bubble.t, bubble.log_y
    wins = [(s, len(t)) for s in range(0, 100, 2)]
    res = lppls_grid_search(t, y, wins, device="cuda", n_tc=40, n_m=20, n_omega=20)
    assert res.tc.shape == (50, 1)
    assert np.isfinite(res.sse).all()
