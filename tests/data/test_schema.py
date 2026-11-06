import numpy as np
import pandas as pd
import pandera.errors as pe
import pytest

from kairos.data.schema import SERIES_COLUMNS, make_series, month_start, validate_series
from kairos.types import SeriesMeta

META = SeriesMeta(
    series_id="c.test",
    name="Test",
    unit="FLOP",
    source="unit",
    source_group="unit",
    side="C",
    transform="log",
    license="CC0-1.0",
    url="https://example.org",
)


def series(**kw):
    return make_series(META, ["2020-01-15", "2020-02-03"], [1.0, 2.0], vintage="2026-11-01", **kw)


def test_make_series_fills_all_columns():
    df = series()
    assert list(df.columns) == list(SERIES_COLUMNS)
    assert df["ds"].tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert (df["side"] == "C").all()
    assert df["y_lo"].isna().all()


def test_bounds_must_bracket_y():
    with pytest.raises(pe.SchemaErrors):
        series(y_lo=[0.5, 3.0], y_hi=[2.0, 4.0])


def test_bad_side_rejected():
    df = series()
    df["side"] = "X"
    with pytest.raises(pe.SchemaErrors):
        validate_series(df)


def test_duplicates_rejected():
    df = pd.concat([series(), series()])
    with pytest.raises(pe.SchemaErrors):
        validate_series(df)


def test_non_finite_rejected():
    df = series()
    df.loc[0, "y"] = np.inf
    with pytest.raises(pe.SchemaErrors):
        validate_series(df)


def test_positive_flag():
    df = series()
    df.loc[0, "y"] = 0.0
    validate_series(df)
    with pytest.raises(ValueError, match="positive"):
        validate_series(df, positive=True)


def test_extra_columns_kept():
    df = make_series(META, ["2020-01-01"], [1.0], vintage="2026-01-01", extra={"gap": [True]})
    assert df["gap"].tolist() == [True]


def test_month_start_tz():
    out = month_start(pd.Series(pd.to_datetime(["2021-03-31T23:00:00Z"])))
    assert out.iloc[0] == pd.Timestamp("2021-03-01")
    assert out.dtype == "datetime64[ns]"
