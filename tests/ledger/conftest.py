"""Shared fixtures for the ledger tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from kairos.ledger.forecast import Forecast, ForecastPoint

ISSUED = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def add_months(d: date, n: int) -> date:
    k = d.year * 12 + d.month - 1 + n
    return date(k // 12, k % 12 + 1, 1)


def build_forecast(
    *,
    series_id: str = "frontier_compute",
    as_of: date = date(2026, 10, 1),
    base: float = 100.0,
    months: int = 6,
    growth: float = 1.1,
    issued_at: datetime = ISSUED,
    **extra: object,
) -> Forecast:
    points = []
    for i in range(1, months + 1):
        mid = base * growth**i
        points.append(
            ForecastPoint(
                ds=add_months(as_of, i), quantiles={0.05: mid * 0.8, 0.5: mid, 0.95: mid * 1.25}
            )
        )
    return Forecast.model_validate(
        {
            "series_id": series_id,
            "model": "prophet-null",
            "as_of": as_of,
            "issued_at": issued_at,
            "unit": "FLOP",
            "points": tuple(points),
            "code_version": "test",
            "params": {"changepoint_prior_scale": 0.05},
            **extra,
        }
    )


@pytest.fixture
def make_forecast():
    """Build forecasts; see :func:`build_forecast` for the keyword arguments."""
    return build_forecast
