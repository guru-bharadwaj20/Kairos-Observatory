from __future__ import annotations

import numpy as np
import pytest

from kairos.accel.coverage import coverage_check


def test_small_coverage_run_is_reproducible():
    a = coverage_check(16, n_boot=100, seed=1, n_jobs=1)
    b = coverage_check(16, n_boot=100, seed=1, n_jobs=2)
    assert a.coverage == b.coverage
    np.testing.assert_array_equal(a.lo, b.lo)
    assert a.n_trials == 16
    assert (a.lo < a.hi).all()
    assert 0.0 <= a.mc_se <= 0.2


def test_calibrated_tolerance():
    r = coverage_check(4, n_boot=50, n_jobs=1)
    assert r.calibrated(tol=1.0)


@pytest.mark.slow
def test_90pct_intervals_cover_true_delta_85_to_95pct():
    r = coverage_check(400, delta=1.0, t_c=2022.0, noise=0.1, n_boot=400, seed=0)
    assert r.calibrated(0.05), r.coverage


@pytest.mark.slow
def test_coverage_close_to_nominal_for_moderate_delta():
    r = coverage_check(400, delta=0.5, noise=0.1, n_boot=400, seed=0)
    assert r.coverage >= 0.8, r.coverage
