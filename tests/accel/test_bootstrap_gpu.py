from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from kairos.accel.bootstrap import block_bootstrap
from kairos.accel.bootstrap_gpu import block_bootstrap_torch
from kairos.accel.synthetic import hyperbolic_series


@pytest.fixture(scope="module")
def series():
    return hyperbolic_series(240, delta=0.5, t_c=2022.0, noise=0.1, ar=0.3, seed=11)


@pytest.fixture(scope="module")
def cpu_reference(series):
    return block_bootstrap(series.t, series.log_y, n_boot=1000, seed=1, n_jobs=1)


def _matches(ref, other):
    assert other.point.delta == pytest.approx(ref.point.delta)
    ks = stats.ks_2samp(ref.delta, other.delta)
    assert ks.pvalue > 0.01, ks
    ks_tc = stats.ks_2samp(ref.t_c, other.t_c)
    assert ks_tc.pvalue > 0.01, ks_tc
    assert other.interval(0.9) == pytest.approx(ref.interval(0.9), abs=0.02)


def test_torch_cpu_float64_matches_reference_distribution(series, cpu_reference):
    res = block_bootstrap_torch(series.t, series.log_y, n_boot=20_000, seed=2, device="cpu")
    assert res.method == "torch:cpu"
    assert res.n_boot == 20_000
    _matches(cpu_reference, res)


def test_torch_chunking_and_seed(series):
    a = block_bootstrap_torch(
        series.t, series.log_y, n_boot=1000, seed=5, device="cpu", max_chunk=300
    )
    b = block_bootstrap_torch(series.t, series.log_y, n_boot=1000, seed=5, device="cpu")
    assert a.n_boot == 1000
    assert np.isfinite(a.delta).all()
    # different chunking draws a different stream, but the same distribution
    assert stats.ks_2samp(a.delta, b.delta, method="asymp").pvalue > 0.01


def test_torch_rejects_bad_args(series):
    with pytest.raises(ValueError, match="n_boot"):
        block_bootstrap_torch(series.t, series.log_y, n_boot=0, device="cpu")
    with pytest.raises(ValueError, match="block_len"):
        block_bootstrap_torch(series.t, series.log_y, n_boot=5, block_len=0, device="cpu")


@pytest.mark.gpu
def test_gpu_float32_1e5_resamples_match_cpu(series, cpu_reference):
    res = block_bootstrap_torch(series.t, series.log_y, n_boot=100_000, seed=3, device="cuda")
    assert res.method.startswith("torch:cuda")
    assert res.n_boot == 100_000
    _matches(cpu_reference, res)
