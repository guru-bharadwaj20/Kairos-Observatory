from __future__ import annotations

import numpy as np
import pytest

from kairos.accel.bootstrap import (
    block_bootstrap,
    block_bootstrap_frame,
    default_block_len,
    mbb_indices,
)
from kairos.accel.synthetic import exponential_series, hyperbolic_series


def test_mbb_indices_are_consecutive_blocks():
    idx = mbb_indices(20, 4, 50, np.random.default_rng(0))
    assert idx.shape == (50, 20)
    assert idx.min() >= 0
    assert idx.max() <= 19
    blocks = idx.reshape(50, 5, 4)
    assert (np.diff(blocks, axis=2) == 1).all()


def test_mbb_truncates_last_block():
    idx = mbb_indices(10, 3, 5, np.random.default_rng(1))
    assert idx.shape == (5, 10)


def test_default_block_len():
    assert default_block_len(240) == 6
    assert default_block_len(5) == 2


def test_bootstrap_reproducible_and_independent_of_workers():
    s = hyperbolic_series(120, delta=0.5, t_c=2012.0, noise=0.03, ar=0.5, seed=1)
    a = block_bootstrap(s.t, s.log_y, n_boot=100, seed=3, n_jobs=1, draws_per_task=30)
    b = block_bootstrap(s.t, s.log_y, n_boot=100, seed=3, n_jobs=2, draws_per_task=30)
    np.testing.assert_array_equal(a.delta, b.delta)
    np.testing.assert_array_equal(a.t_c, b.t_c)
    c = block_bootstrap(s.t, s.log_y, n_boot=100, seed=4, n_jobs=1)
    assert not np.array_equal(a.delta, c.delta)


def test_bootstrap_interval_covers_truth():
    s = hyperbolic_series(240, delta=0.5, t_c=2022.0, noise=0.1, ar=0.3, seed=2)
    res = block_bootstrap(s.t, s.log_y, n_boot=400, seed=0, n_jobs=1)
    assert res.n_boot == 400
    lo, hi = res.interval(0.9)
    assert lo < 0.5 < hi
    assert res.se() > 0
    tlo, thi = res.interval(0.9, "t_c")
    assert tlo < 2022.0 < thi
    assert res.frac_singular == 1.0
    assert res.p_delta_positive == pytest.approx(1 / 401)
    rlo, _ = res.interval(0.9, "t_remaining")
    assert rlo == pytest.approx(tlo - s.t[-1])


def test_exponential_has_unbounded_tc_interval():
    s = exponential_series(240, noise=0.05, ar=0.5, seed=1)
    res = block_bootstrap(s.t, s.log_y, n_boot=200, seed=0, n_jobs=1)
    lo, hi = res.interval(0.9)
    assert lo < 0 < hi
    assert np.isinf(res.interval(0.9, "t_c")[1])
    assert res.p_delta_positive > 0.05


def test_reselect_lambda_path():
    s = hyperbolic_series(96, delta=0.5, t_c=2010.0, noise=0.03, seed=3)
    res = block_bootstrap(s.t, s.log_y, n_boot=8, seed=0, n_jobs=1, reselect_lambda=True)
    assert np.isfinite(res.delta).all()


def test_frame_wrapper_and_validation():
    s = hyperbolic_series(96, delta=0.5, t_c=2010.0, seed=3)
    res = block_bootstrap_frame(s.frame, n_boot=10, seed=0, n_jobs=1)
    assert res.n_boot == 10
    with pytest.raises(ValueError, match="level"):
        res.interval(1.5)
    with pytest.raises(ValueError, match="n_boot"):
        block_bootstrap(s.t, s.log_y, n_boot=0)
