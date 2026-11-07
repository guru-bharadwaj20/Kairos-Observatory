from __future__ import annotations

import pytest

from kairos.accel.lppls import fit_lppls, lppls_series
from kairos.accel.lppls_gpu import lppls_grid_search
from kairos.accel.lppls_refine import refine_grid


@pytest.fixture(scope="module")
def bubble():
    return lppls_series(140, start="2009-01-01", tc=2021.0, noise=0.005, seed=4)


def test_refinement_never_worse_than_grid_and_matches_reference(bubble):
    t, y = bubble.t, bubble.log_y
    wins = [(0, len(t)), (20, len(t))]
    grid = lppls_grid_search(t, y, wins, device="cpu", top_k=3, n_tc=12, n_m=8, n_omega=8)
    fits = refine_grid(t, y, grid, n_jobs=1)
    assert len(fits) == 2
    for w, fit in enumerate(fits):
        assert fit.sse <= grid.sse[w, 0] * (1 + 1e-9)
    ref = fit_lppls(t, y)
    assert fits[0].sse == pytest.approx(ref.sse, rel=0.05)
    assert fits[0].tc == pytest.approx(2021.0, abs=0.2)


def test_parallel_matches_serial(bubble):
    t, y = bubble.t, bubble.log_y
    grid = lppls_grid_search(t, y, [(0, len(t))], device="cpu", top_k=2, n_tc=6, n_m=4, n_omega=4)
    a = refine_grid(t, y, grid, n_jobs=1)
    b = refine_grid(t, y, grid, n_jobs=2)
    assert a[0].sse == pytest.approx(b[0].sse)


@pytest.mark.gpu
def test_refines_gpu_cells(bubble):
    t, y = bubble.t, bubble.log_y
    grid = lppls_grid_search(t, y, [(0, len(t))], device="cuda", top_k=4)
    (fit,) = refine_grid(t, y, grid, n_jobs=1)
    assert fit.tc == pytest.approx(2021.0, abs=0.2)
