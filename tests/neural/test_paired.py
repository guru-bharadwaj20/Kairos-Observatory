from __future__ import annotations

import numpy as np
import pytest

from kairos.neural import NeuralConfig, synthetic_series
from kairos.neural.paired import (
    SELF_COLUMN,
    FitTask,
    PairedSpec,
    _variant_frame,
    block_shuffled,
    fit_task,
    paired_tasks,
    run_paired,
)
from kairos.neural.regressors import design_frame

CFG = NeuralConfig(n_forecasts=3, epochs=15, early_stopping=False)
SPEC = PairedSpec(config=CFG, self_lags=6, test_months=12)


def _design(coupling=0.9, n=96):
    return design_frame(synthetic_series(n, coupling=coupling, seed=0), "capability")


def test_spec_rejects_ar_lags():
    with pytest.raises(ValueError, match="n_lags"):
        PairedSpec(config=NeuralConfig(n_lags=3))


def test_block_shuffled_keeps_increments():
    x = np.cumsum(np.arange(1.0, 25.0))
    s = block_shuffled(x, 4, seed=1)
    assert s[0] == x[0]
    assert s[-1] == pytest.approx(x[-1])
    np.testing.assert_allclose(np.sort(np.diff(s)), np.sort(np.diff(x)))
    assert not np.allclose(s, x)
    np.testing.assert_array_equal(s, block_shuffled(x, 4, seed=1))


def test_variant_frames_differ_only_in_self_channel():
    design = _design()
    exo = _variant_frame(FitTask(design, SPEC, "exo", 0))
    self_ = _variant_frame(FitTask(design, SPEC, "self", 0))
    null = _variant_frame(FitTask(design, SPEC, "null", 0, perm=0, perm_seed=3))
    assert SELF_COLUMN not in exo.columns
    np.testing.assert_array_equal(self_[SELF_COLUMN], self_["y"])
    assert not np.allclose(null[SELF_COLUMN], null["y"])
    assert exo.equals(self_.drop(columns=SELF_COLUMN))


def test_paired_tasks_share_seed_and_data():
    tasks = paired_tasks(_design(), SPEC, [1, 2])
    assert [(t.variant, t.seed) for t in tasks] == [
        ("exo", 1),
        ("self", 1),
        ("exo", 2),
        ("self", 2),
    ]
    assert all(t.design is tasks[0].design for t in tasks)


def test_run_paired_scores_same_holdout():
    exo, self_ = run_paired(_design(), SPEC, seed=0)
    assert len(exo.errors) == len(self_.errors) == 3
    assert exo.mae > 0
    assert self_.mae > 0


def test_null_fit_runs():
    result = fit_task(FitTask(_design(), SPEC, "null", 0, perm=2, perm_seed=5))
    assert result.perm == 2
    assert np.isfinite(result.mae)
