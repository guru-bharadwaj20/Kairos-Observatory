from __future__ import annotations

import pandas as pd
import pytest

from kairos.neural import NeuralConfig, synthetic_series
from kairos.neural.global_model import compare_global_local, fit_global, global_frame

CFG = NeuralConfig(
    n_lags=4,
    n_forecasts=2,
    epochs=10,
    early_stopping=False,
    n_changepoints=0,
    trend_global_local="local",
)


def _frame(n=60):
    parts = [synthetic_series(n, seed=i, drivers={}, series_id=f"lab{i}") for i in range(3)]
    return pd.concat(parts, ignore_index=True)


def test_global_frame_layout():
    df = global_frame(_frame(), ["lab0", "lab1"])
    assert list(df.columns) == ["ds", "y", "ID"]
    assert df.groupby("ID").size().to_dict() == {"lab0": 60, "lab1": 60}


def test_fit_global_scores_each_series():
    df = global_frame(_frame(), ["lab0", "lab1", "lab2"])
    scores = fit_global(df, CFG, test_months=12, device="cpu", seed=0)
    assert set(scores) == {"lab0", "lab1", "lab2"}
    assert all(v > 0 for v in scores.values())


def test_short_series_rejected():
    df = global_frame(_frame(10), ["lab0"])
    with pytest.raises(ValueError, match="shorter"):
        fit_global(df, CFG, test_months=12, device="cpu")


def test_compare_global_local():
    df = global_frame(_frame(), ["lab0", "lab1"])
    table = compare_global_local(df, CFG, test_months=12, device="cpu", seed=0, n_jobs=1)
    assert list(table["series"]) == ["lab0", "lab1"]
    assert set(table["better"]) <= {"global", "local"}
    assert table["mae_local"].notna().all()


@pytest.mark.gpu
def test_global_on_gpu():
    df = global_frame(_frame(), ["lab0", "lab1", "lab2"])
    assert len(fit_global(df, CFG, test_months=12, device="cuda", seed=0)) == 3
