from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from kairos.accel.bootstrap import block_bootstrap
from kairos.accel.multitest import benjamini_hochberg, delta_significance
from kairos.accel.synthetic import exponential_series, hyperbolic_series


def test_bh_matches_scipy():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=40), rng.uniform(0, 0.005, size=10)])
    reject, adj = benjamini_hochberg(p, 0.1)
    np.testing.assert_allclose(adj, stats.false_discovery_control(p, method="bh"))
    assert reject.sum() >= 10
    np.testing.assert_array_equal(reject, adj <= 0.1)


def test_bh_textbook_example():
    p = [0.01, 0.04, 0.03, 0.005]
    reject, adj = benjamini_hochberg(p, 0.05)
    assert reject.all()
    np.testing.assert_allclose(adj, [0.02, 0.04, 0.04, 0.02])


def test_bh_handles_nan_and_validates():
    reject, adj = benjamini_hochberg([0.01, np.nan], 0.05)
    assert reject.tolist() == [True, False]
    assert np.isnan(adj[1])
    with pytest.raises(ValueError, match="q"):
        benjamini_hochberg([0.1], 0.0)
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg([1.5])


def test_delta_significance_separates_hyperbolic_from_exponential():
    results = {}
    for i in range(3):
        s = hyperbolic_series(180, delta=0.5, t_c=2017.0, noise=0.05, seed=i)
        results[f"hyp{i}"] = block_bootstrap(s.t, s.log_y, n_boot=199, seed=i, n_jobs=1)
        e = exponential_series(180, noise=0.05, seed=10 + i)
        results[f"exp{i}"] = block_bootstrap(e.t, e.log_y, n_boot=199, seed=i, n_jobs=1)
    df = delta_significance(results, q=0.1)
    sig = set(df.loc[df["significant"], "series_id"])
    assert {"hyp0", "hyp1", "hyp2"} <= sig
    assert len(sig) <= 4  # FDR control leaves at most a stray exponential
    assert list(df.columns) == [
        "series_id",
        "delta",
        "delta_lo",
        "delta_hi",
        "p_value",
        "p_adj",
        "significant",
    ]
