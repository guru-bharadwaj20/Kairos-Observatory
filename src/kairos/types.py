"""Core shared types.

``SeriesFrame`` convention
--------------------------
Every time series moves through KAIROS as a long-format :class:`pandas.DataFrame`:

========== ================== ==================================================
column     dtype              meaning
========== ================== ==================================================
``ds``     datetime64[ns]     month start (``freq="MS"``), tz-naive
``y``      float64            value in **raw positive units** (not logged)
``series_id`` str             stable identifier, matches :class:`SeriesMeta`
``source`` str (optional)     provider, e.g. ``"epoch"``
``source_group`` str (opt.)   provenance cluster for independence (D10)
``unit``   str (optional)     unit of ``y``, e.g. ``"FLOP"``
``y_lo``   float (optional)   lower uncertainty bound, raw units
``y_hi``   float (optional)   upper uncertainty bound, raw units
========== ================== ==================================================

Transforms (``log``/``logit``) are applied by models, never stored in ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Literal, TypeAlias

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "DialReading",
    "SeriesFrame",
    "SeriesMeta",
    "Side",
    "Transform",
    "validate_series_frame",
]

SeriesFrame: TypeAlias = pd.DataFrame  # noqa: UP040 (a runtime alias of DataFrame)
"""A DataFrame following the ``SeriesFrame`` convention in this module's docstring."""

Side = Literal["C", "H", "exo", "validation"]
Transform = Literal["log", "logit", "none"]

REQUIRED_COLUMNS: Final = ("ds", "y", "series_id")
OPTIONAL_COLUMNS: Final = ("source", "source_group", "unit", "y_lo", "y_hi")


@dataclass(frozen=True, slots=True)
class SeriesMeta:
    """Metadata for one series.

    Attributes:
        series_id: Stable identifier used in ``SeriesFrame.series_id``.
        name: Human-readable name.
        unit: Unit of the raw values.
        source: Provider key (``"epoch"``, ``"metr"``, ...).
        source_group: Provenance cluster; different groups count as independent (D10).
        side: ``"C"`` capability, ``"H"`` human capacity, ``"exo"`` driver or
            ``"validation"`` historical test case.
        transform: Transform models apply before fitting.
        license: SPDX identifier or licence name of the source data.
        url: Where the source data lives.
    """

    series_id: str
    name: str
    unit: str
    source: str
    source_group: str
    side: Side
    transform: Transform
    license: str
    url: str


def validate_series_frame(df: pd.DataFrame, *, positive: bool = True) -> SeriesFrame:
    """Check that ``df`` follows the ``SeriesFrame`` convention and return it.

    Args:
        df: Frame to check.
        positive: Require ``y > 0`` (true for log-transformed series).

    Raises:
        ValueError: If a required column is missing or a value breaks the convention.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"SeriesFrame is missing columns: {missing}")
    if not pd.api.types.is_datetime64_ns_dtype(df["ds"]):
        raise ValueError(f"ds must be datetime64[ns], got {df['ds'].dtype}")
    if not df["ds"].dt.is_month_start.all():
        raise ValueError("ds must contain month-start dates")
    if not pd.api.types.is_float_dtype(df["y"]):
        raise ValueError(f"y must be float, got {df['y'].dtype}")
    if positive and not (df["y"].dropna() > 0).all():
        raise ValueError("y must be strictly positive (raw units)")
    if df.duplicated(["series_id", "ds"]).any():
        raise ValueError("duplicate (series_id, ds) rows")
    return df


class DialReading(BaseModel):
    """One reading of one KAIROS dial.

    Attributes:
        dial: Dial name (``"acceleration"``, ``"recursion"``, ``"vinge"``, ``"jonas"``).
        value: Point estimate.
        level: Tripwire contribution, 0 (Normal) to 4 (Kairos).
        ci_lo: Lower bound of the reported interval.
        ci_hi: Upper bound of the reported interval.
        as_of: Data vintage date of the reading.
        detail: Free-form diagnostics (per-metric values, p-values, ...).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dial: str
    value: float
    level: int = Field(ge=0, le=4)
    ci_lo: float
    ci_hi: float
    as_of: date
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ordered_interval(self) -> DialReading:
        if self.ci_lo > self.ci_hi:
            raise ValueError("ci_lo must be <= ci_hi")
        return self
