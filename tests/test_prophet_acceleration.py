import numpy as np
import pandas as pd
import pytest

from kairos.models.acceleration import RUN_COLUMNS, acceleration_runs
from kairos.models.changepoints import changepoint_table
from kairos.models.null_model import NullModelFit, fit_null_model
from kairos.models.synthetic import piecewise_exponential


def _fit(deltas, sid="a", base=0.2):
    return NullModelFit(
        series_id=sid,
        history=pd.DataFrame(),
        forecast=pd.DataFrame(),
        changepoints=pd.date_range("2000-01-01", periods=len(deltas), freq="12MS"),
        deltas=np.asarray(deltas, dtype="float64"),
        base_rate=base,
        frontier=False,
        seed=0,
        fit_seconds=0.0,
    )


def test_runs_skip_insignificant_and_break_on_decrease():
    # +, (tiny), +, +, -, +, +  -> runs [0,2,3] and [5,6]
    table = changepoint_table(_fit([0.1, 0.001, 0.2, 0.3, -0.1, 0.05, 0.05]))
    runs = acceleration_runs(table)
    assert tuple(runs.columns) == RUN_COLUMNS
    assert list(runs["n_changepoints"]) == [3, 2]
    assert runs["total_delta"].iloc[0] == pytest.approx(0.6)
    assert runs["rate_start"].iloc[0] == pytest.approx(0.2)
    assert runs["rate_end"].iloc[0] == pytest.approx(0.801)
    assert runs["rate_ratio"].iloc[0] == pytest.approx(0.801 / 0.2)
    assert list(runs["ongoing"]) == [False, True]


def test_min_run_and_series_separation():
    table = changepoint_table({"a": _fit([0.1, -0.1, 0.1], "a"), "b": _fit([0.1, 0.1], "b")})
    runs = acceleration_runs(table, min_run=2)
    assert list(runs["series_id"]) == ["b"]
    assert len(acceleration_runs(table, min_run=1)) == 3
    with pytest.raises(ValueError, match="min_run"):
        acceleration_runs(table, min_run=0)


def test_no_runs_gives_empty_frame():
    runs = acceleration_runs(changepoint_table(_fit([-0.1, -0.2])))
    assert runs.empty
    assert tuple(runs.columns) == RUN_COLUMNS


def test_detects_super_exponential_segment():
    # Growth rate stepping up every two years from 2012: a piecewise hyperbola.
    breaks = [f"{y}-01-01" for y in range(2012, 2022, 2)]
    rates = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7]
    frame = piecewise_exponential("2000-01-01", "2024-12-01", breaks=breaks, rates=rates)
    fit = fit_null_model(frame, seed=1, horizon=0)
    runs = acceleration_runs(changepoint_table(fit, min_magnitude=0.05))
    assert not runs.empty
    best = runs.sort_values("total_delta").iloc[-1]
    assert best["total_delta"] == pytest.approx(1.5, abs=0.3)
    assert pd.Timestamp("2010-01-01") <= best["start"] <= pd.Timestamp("2014-01-01")
