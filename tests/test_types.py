from datetime import date

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from kairos.types import DialReading, SeriesMeta, validate_series_frame


def frame(n=12, y=None):
    return pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=n, freq="MS"),
            "y": np.linspace(1.0, 2.0, n) if y is None else y,
            "series_id": "s",
        }
    )


def test_valid_frame_passes():
    df = frame()
    assert validate_series_frame(df) is df


def test_missing_column():
    with pytest.raises(ValueError, match="missing"):
        validate_series_frame(frame().drop(columns="series_id"))


def test_not_month_start():
    df = frame()
    df.loc[0, "ds"] = pd.Timestamp("2020-01-15")
    with pytest.raises(ValueError, match="month-start"):
        validate_series_frame(df)


def test_non_positive():
    with pytest.raises(ValueError, match="positive"):
        validate_series_frame(frame(3, y=[1.0, 0.0, 2.0]))
    validate_series_frame(frame(3, y=[1.0, 0.0, 2.0]), positive=False)


def test_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        validate_series_frame(pd.concat([frame(), frame()]))


def test_series_meta_is_frozen():
    meta = SeriesMeta("c", "Compute", "FLOP", "epoch", "epoch", "C", "log", "CC-BY-4.0", "u")
    with pytest.raises(AttributeError):
        meta.name = "x"


@given(level=st.integers(0, 4), lo=st.floats(-1e6, 0), hi=st.floats(0, 1e6))
def test_dial_reading_accepts_valid(level, lo, hi):
    r = DialReading(
        dial="vinge", value=0.0, level=level, ci_lo=lo, ci_hi=hi, as_of=date(2026, 11, 6)
    )
    assert r.level == level


@pytest.mark.parametrize("bad", [{"level": 5}, {"level": -1}, {"ci_lo": 2.0, "ci_hi": 1.0}])
def test_dial_reading_rejects_invalid(bad):
    base = {
        "dial": "jonas",
        "value": 1.0,
        "level": 1,
        "ci_lo": 0.0,
        "ci_hi": 2.0,
        "as_of": date(2026, 1, 1),
    }
    with pytest.raises(ValidationError):
        DialReading(**{**base, **bad})
