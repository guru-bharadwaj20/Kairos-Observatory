"""The forecast artifact that gets pre-registered, and its content hash.

A :class:`Forecast` is a quantile forecast of one series issued by one model at a
data vintage ``as_of``. Its identity is :func:`forecast_hash`: SHA-256 of its
canonical JSON (:mod:`kairos.ledger.canonical`), so the same forecast always gets
the same hash, on any machine.
"""

from __future__ import annotations

import itertools
from datetime import date, datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kairos.ledger.canonical import sha256_hex
from kairos.types import DialReading

__all__ = [
    "SCHEMA_VERSION",
    "Forecast",
    "ForecastPoint",
    "forecast_hash",
    "quantile_key",
]

SCHEMA_VERSION: Final = 1


def quantile_key(level: float) -> str:
    """Return the canonical string key for quantile ``level`` (``0.05`` -> ``"0.05"``)."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"quantile level must be in (0, 1), got {level}")
    return f"{round(level, 6):g}"


class ForecastPoint(BaseModel):
    """Forecast quantiles for one future date.

    Attributes:
        ds: Target date (month start for monthly series).
        quantiles: Quantile level (as a canonical string such as ``"0.05"``) to
            value in raw units. Values must be non-decreasing in the level.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ds: date
    quantiles: dict[str, float] = Field(min_length=1)

    @field_validator("quantiles", mode="before")
    @classmethod
    def _normalise_keys(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {quantile_key(float(k)): v for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def _monotone(self) -> ForecastPoint:
        ordered = sorted(self.quantiles.items(), key=lambda kv: float(kv[0]))
        values = [v for _, v in ordered]
        if any(b < a for a, b in itertools.pairwise(values)):
            raise ValueError(f"quantiles must be non-decreasing at {self.ds}: {ordered}")
        return self

    def levels(self) -> list[float]:
        """Return the quantile levels in increasing order."""
        return sorted(float(k) for k in self.quantiles)

    def at(self, level: float) -> float:
        """Return the value of quantile ``level``."""
        return self.quantiles[quantile_key(level)]


class Forecast(BaseModel):
    """A pre-registrable quantile forecast.

    Attributes:
        schema_version: Record format version.
        series_id: Series being forecast (see :class:`kairos.types.SeriesMeta`).
        model: Model name, e.g. ``"prophet-null"`` or ``"neuralprophet-ar"``.
        as_of: Data vintage the forecast was fitted on (no lookahead past it).
        issued_at: When the forecast was produced (timezone-aware).
        unit: Unit of the forecast values.
        points: One :class:`ForecastPoint` per horizon step, strictly increasing ``ds``.
        code_version: Git revision or package version of the code that produced it.
        params: Model parameters needed to reproduce the forecast.
        dials: Dial readings published alongside the forecast.
        notes: Free text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    series_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    as_of: date
    issued_at: datetime
    unit: str = ""
    points: tuple[ForecastPoint, ...] = Field(min_length=1)
    code_version: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    dials: tuple[DialReading, ...] = ()
    notes: str = ""

    @field_validator("issued_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("issued_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _ordered_points(self) -> Forecast:
        dates = [p.ds for p in self.points]
        if any(b <= a for a, b in itertools.pairwise(dates)):
            raise ValueError("forecast points must have strictly increasing ds")
        if dates[0] <= self.as_of:
            raise ValueError("forecast points must lie after as_of")
        return self

    @property
    def horizon_end(self) -> date:
        """Last target date of the forecast."""
        return self.points[-1].ds

    def payload(self) -> dict[str, Any]:
        """Return the record as plain Python data (what the ledger stores)."""
        return self.model_dump(mode="json", exclude_none=True)


def forecast_hash(forecast: Forecast | dict[str, Any]) -> str:
    """Return the SHA-256 content hash of a forecast (or of its stored payload).

    A forecast and the payload read back from the ledger hash identically.
    """
    if isinstance(forecast, dict):
        forecast = Forecast.model_validate(forecast)
    return sha256_hex(forecast)
