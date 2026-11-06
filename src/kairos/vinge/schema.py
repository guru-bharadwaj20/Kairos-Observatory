"""Input contract between the forecast phases (3, 4, 5) and the Vinge Horizon.

The Vinge Horizon never imports a concrete model. Anything that satisfies
:class:`Forecaster` (Prophet null models, NeuralProphet, a naive baseline) can be
backtested, and anything that produces a :data:`BacktestFrame` (for example
Prophet's own ``cross_validation`` output, see
:func:`kairos.vinge.backtest.from_prophet_cv`) can be scored.

``ForecastFrame`` (what a :class:`Forecaster` returns)
------------------------------------------------------
============ ============== ====================================================
column       dtype          meaning
============ ============== ====================================================
``ds``       datetime64     target month (month start), strictly after the cutoff
``yhat``     float          point forecast (median), **raw positive units**
``yhat_lo``  float          lower bound of the central interval, raw units
``yhat_hi``  float          upper bound of the central interval, raw units
``q_<p>``    float (opt.)   extra quantiles, e.g. ``q_0.05``; ``p`` in (0, 1)
============ ============== ====================================================

``BacktestFrame`` (one row per series x cutoff x target month)
---------------------------------------------------------------
All ``ForecastFrame`` columns plus ``series_id`` (str), ``cutoff`` (datetime64,
last month of data the model saw), ``horizon`` (int, months after the cutoff,
``>= 1``) and ``y`` (float, the realised value in raw units).
"""

from __future__ import annotations

import re
from typing import Final, Protocol, TypeAlias, runtime_checkable

import pandas as pd

from kairos.types import SeriesFrame

__all__ = [
    "BACKTEST_COLUMNS",
    "DEFAULT_INTERVAL",
    "FORECAST_COLUMNS",
    "BacktestFrame",
    "ForecastFrame",
    "Forecaster",
    "quantile_columns",
    "validate_backtest_frame",
    "validate_forecast_frame",
]

ForecastFrame: TypeAlias = pd.DataFrame  # noqa: UP040 (runtime alias)
BacktestFrame: TypeAlias = pd.DataFrame  # noqa: UP040 (runtime alias)

FORECAST_COLUMNS: Final = ("ds", "yhat", "yhat_lo", "yhat_hi")
BACKTEST_COLUMNS: Final = ("series_id", "cutoff", "ds", "horizon", "y", *FORECAST_COLUMNS[1:])
DEFAULT_INTERVAL: Final = 0.90
"""Coverage of the ``yhat_lo``/``yhat_hi`` band unless a forecaster says otherwise."""

_QUANTILE = re.compile(r"^q_(0?\.\d+)$")


@runtime_checkable
class Forecaster(Protocol):
    """A model the backtest harness can refit at every cutoff.

    Implementations must be picklable (the harness runs them in worker
    processes) and deterministic for a given input (fits are cached by content).
    """

    @property
    def interval_width(self) -> float:
        """Nominal coverage of ``yhat_lo``/``yhat_hi``, e.g. ``0.90``."""
        ...

    def fit_predict(self, history: SeriesFrame, horizon: int) -> ForecastFrame:
        """Fit on ``history`` (one series, rows up to the cutoff) and forecast.

        Args:
            history: ``SeriesFrame`` rows of a single series, already truncated
                to what was known at the cutoff.
            horizon: Number of monthly steps to forecast after the last ``ds``.

        Returns:
            A ``ForecastFrame`` with exactly ``horizon`` rows.
        """
        ...


def quantile_columns(df: pd.DataFrame) -> dict[float, str]:
    """Return ``{level: column}`` for every ``q_<p>`` column, sorted by level."""
    found: dict[float, str] = {}
    for col in df.columns:
        match = _QUANTILE.match(str(col))
        if match:
            level = float(match.group(1))
            if not 0.0 < level < 1.0:
                raise ValueError(f"quantile level out of (0, 1): {col}")
            found[level] = str(col)
    return dict(sorted(found.items()))


def _check_positive(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    for col in cols:
        values = df[col].dropna()
        if not (values > 0).all():
            raise ValueError(f"{col} must be strictly positive (raw units)")


def validate_forecast_frame(df: pd.DataFrame, horizon: int | None = None) -> ForecastFrame:
    """Check a ``ForecastFrame`` and return it.

    Raises:
        ValueError: On a missing column, wrong length, non-positive value or
            ``yhat_lo > yhat_hi``.
    """
    missing = [c for c in FORECAST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ForecastFrame is missing columns: {missing}")
    if horizon is not None and len(df) != horizon:
        raise ValueError(f"expected {horizon} forecast rows, got {len(df)}")
    _check_positive(df, ("yhat", "yhat_lo", "yhat_hi", *quantile_columns(df).values()))
    if (df["yhat_lo"] > df["yhat_hi"]).any():
        raise ValueError("yhat_lo must be <= yhat_hi")
    return df


def validate_backtest_frame(df: pd.DataFrame) -> BacktestFrame:
    """Check a ``BacktestFrame`` and return it.

    Raises:
        ValueError: On a missing column, a non-positive horizon or value, or a
            target month that is not after its cutoff (lookahead).
    """
    missing = [c for c in BACKTEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"BacktestFrame is missing columns: {missing}")
    validate_forecast_frame(df)
    _check_positive(df, ("y",))
    if (df["horizon"] < 1).any():
        raise ValueError("horizon must be >= 1")
    if (df["ds"] <= df["cutoff"]).any():
        raise ValueError("every target ds must be after its cutoff")
    return df
