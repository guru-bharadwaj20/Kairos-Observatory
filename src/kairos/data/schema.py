"""Pandera schema for stored KAIROS series.

A *stored* series is a :data:`~kairos.types.SeriesFrame` with every optional column
filled in, plus ``side`` and ``vintage``:

================ ================ ===============================================
column           dtype            meaning
================ ================ ===============================================
``ds``           datetime64[ns]   month start, tz-naive
``y``            float64          value in raw units (finite)
``series_id``    str              matches :class:`~kairos.types.SeriesMeta`
``side``         str              ``C``, ``H``, ``exo`` or ``validation``
``unit``         str              unit of ``y``
``source``       str              provider key
``source_group`` str              provenance cluster (D10)
``vintage``      datetime64[ns]   date the source data were retrieved
``y_lo``         float64          lower bound (NaN when unknown)
``y_hi``         float64          upper bound (NaN when unknown)
================ ================ ===============================================

Extra columns (for example ``interpolated`` or ``gap``) are allowed and kept.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any, Final, get_args

import numpy as np
import pandas as pd
import pandera.pandas as pa

from kairos.types import SeriesFrame, SeriesMeta, Side, validate_series_frame

__all__ = [
    "SERIES_COLUMNS",
    "SERIES_SCHEMA",
    "SIDES",
    "make_series",
    "month_start",
    "validate_series",
]

SIDES: Final[tuple[str, ...]] = get_args(Side)
SERIES_COLUMNS: Final = (
    "ds",
    "y",
    "series_id",
    "side",
    "unit",
    "source",
    "source_group",
    "vintage",
    "y_lo",
    "y_hi",
)


def _bounds_ordered(df: pd.DataFrame) -> pd.Series:
    lo_ok = df["y_lo"].isna() | (df["y_lo"] <= df["y"])
    hi_ok = df["y_hi"].isna() | (df["y_hi"] >= df["y"])
    return lo_ok & hi_ok


SERIES_SCHEMA: Final = pa.DataFrameSchema(
    {
        "ds": pa.Column(
            "datetime64[ns]",
            pa.Check(lambda s: s.dt.is_month_start, error="ds must be a month start"),
        ),
        "y": pa.Column(
            float, pa.Check(lambda s: np.isfinite(s), error="y must be finite"), nullable=False
        ),
        "series_id": pa.Column(str, pa.Check.str_length(min_value=1)),
        "side": pa.Column(str, pa.Check.isin(SIDES)),
        "unit": pa.Column(str, pa.Check.str_length(min_value=1)),
        "source": pa.Column(str, pa.Check.str_length(min_value=1)),
        "source_group": pa.Column(str, pa.Check.str_length(min_value=1)),
        "vintage": pa.Column("datetime64[ns]"),
        "y_lo": pa.Column(float, nullable=True),
        "y_hi": pa.Column(float, nullable=True),
    },
    checks=[pa.Check(_bounds_ordered, error="need y_lo <= y <= y_hi")],
    unique=["series_id", "ds"],
    strict=False,
    ordered=False,
    name="KairosSeries",
)
"""Schema every processed series file must satisfy."""


def month_start(values: Any) -> pd.Series:
    """Convert dates (strings, timestamps, periods) to tz-naive month starts.

    Args:
        values: Anything :func:`pandas.to_datetime` accepts, as a list or Series.

    Returns:
        A ``datetime64[ns]`` Series of month-start dates.
    """
    ts = pd.Series(pd.to_datetime(values))
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    return ts.dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")


def validate_series(df: pd.DataFrame, *, positive: bool = False) -> SeriesFrame:
    """Validate a stored series against :data:`SERIES_SCHEMA` and the core convention.

    Args:
        df: Frame to check.
        positive: Also require ``y > 0`` (log-transformed series).

    Returns:
        The validated frame (same object, columns coerced where pandera does so).

    Raises:
        pandera.errors.SchemaErrors: If the frame breaks the schema (all failures listed).
        ValueError: If it breaks :func:`kairos.types.validate_series_frame`.
    """
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    out = SERIES_SCHEMA.validate(df, lazy=True)
    return validate_series_frame(out, positive=positive)


def make_series(
    meta: SeriesMeta,
    ds: Iterable[Any],
    y: Iterable[float],
    *,
    vintage: date | str,
    y_lo: Iterable[float] | None = None,
    y_hi: Iterable[float] | None = None,
    extra: dict[str, Iterable[Any]] | None = None,
) -> SeriesFrame:
    """Build a validated stored series from values and metadata.

    Dates are snapped to month starts and the result is sorted by ``ds``.

    Args:
        meta: Series metadata; fills ``series_id``, ``side``, ``unit`` and provenance.
        ds: Observation dates.
        y: Values in raw units.
        vintage: Retrieval date of the source data.
        y_lo: Optional lower bounds.
        y_hi: Optional upper bounds.
        extra: Optional extra columns (name -> values).

    Returns:
        A frame that passes :func:`validate_series`.
    """
    frame = pd.DataFrame({"ds": month_start(list(ds)), "y": np.asarray(list(y), dtype=float)})
    n = len(frame)
    nan = np.full(n, np.nan)
    frame["series_id"] = meta.series_id
    frame["side"] = meta.side
    frame["unit"] = meta.unit
    frame["source"] = meta.source
    frame["source_group"] = meta.source_group
    frame["vintage"] = pd.Timestamp(vintage).normalize().as_unit("ns")
    frame["vintage"] = frame["vintage"].astype("datetime64[ns]")
    frame["y_lo"] = nan if y_lo is None else np.asarray(list(y_lo), dtype=float)
    frame["y_hi"] = nan if y_hi is None else np.asarray(list(y_hi), dtype=float)
    for name, values in (extra or {}).items():
        frame[name] = list(values)
    frame = frame.sort_values("ds", kind="stable").reset_index(drop=True)
    return validate_series(frame)
