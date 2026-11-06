import numpy as np
import pandas as pd
import pytest

from kairos.models.changepoints import CHANGEPOINT_COLUMNS, changepoint_table
from kairos.models.null_model import NullModelFit, fit_null_model
from kairos.models.synthetic import piecewise_exponential


def _fake_fit(sid="a", deltas=(0.0, 0.5, -0.2, 0.005)):
    cps = pd.date_range("2010-01-01", periods=len(deltas), freq="12MS")
    return NullModelFit(
        series_id=sid,
        history=pd.DataFrame(),
        forecast=pd.DataFrame(),
        changepoints=cps,
        deltas=np.asarray(deltas, dtype="float64"),
        base_rate=0.3,
        frontier=False,
        seed=0,
        fit_seconds=0.0,
    )


def test_table_columns_rates_and_flags():
    table = changepoint_table(_fake_fit())
    assert tuple(table.columns) == CHANGEPOINT_COLUMNS
    np.testing.assert_allclose(table["rate_before"], [0.3, 0.3, 0.8, 0.6])
    np.testing.assert_allclose(table["rate_after"], [0.3, 0.8, 0.6, 0.605])
    assert list(table["significant"]) == [False, True, True, False]
    assert list(table["rank"]) == [4, 1, 2, 3]


def test_table_many_fits_and_filter():
    fits = {"a": _fake_fit("a"), "b": _fake_fit("b", (0.1, 0.2))}
    table = changepoint_table(fits, significant_only=True)
    assert set(table["series_id"]) == {"a", "b"}
    assert table["significant"].all()
    assert changepoint_table([]).empty


def test_table_from_real_fit_finds_break():
    frame = piecewise_exponential(
        "2005-01-01", "2024-12-01", breaks=("2015-01-01",), rates=(0.2, 1.0), noise=0.02
    )
    fit = fit_null_model(frame, seed=4, horizon=0)
    table = changepoint_table(fit)
    assert len(table) == 25
    top = table.loc[table["rank"] == 1].iloc[0]
    assert top["delta"] > 0
    assert abs((top["ds"] - pd.Timestamp("2015-01-01")).days) < 365
    assert table["delta"].sum() == pytest.approx(0.8, abs=0.1)
