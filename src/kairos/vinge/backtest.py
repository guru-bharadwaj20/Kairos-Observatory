"""Rolling-origin backtest harness with no lookahead (K-6.01).

Two entry points produce the same :data:`~kairos.vinge.schema.BacktestFrame`:

* :func:`rolling_origin_backtest` refits any :class:`~kairos.vinge.schema.Forecaster`
  at every cutoff, one ``(series, cutoff)`` task per worker via
  :func:`kairos.compute.pmap`, each fit memoised with
  :func:`kairos.compute.cached_fit`. It honours data **vintages**: if the frame
  has a ``vintage`` column (date a value was published), a row is visible at a
  cutoff only when both ``ds`` and ``vintage`` are on or before it.
* :func:`prophet_cross_validation` wraps a fitted Prophet model's
  ``prophet.diagnostics.cross_validation(parallel="processes")`` on explicit
  monthly cutoffs and converts the result with :func:`from_prophet_cv`. Prophet
  truncates by ``ds`` only, so use it for series without revisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from kairos.compute import cached_fit, pmap
from kairos.logging import get_logger
from kairos.types import SeriesFrame
from kairos.vinge.schema import (
    BACKTEST_COLUMNS,
    BacktestFrame,
    Forecaster,
    quantile_columns,
    validate_backtest_frame,
    validate_forecast_frame,
)

__all__ = [
    "VINTAGE_COLUMN",
    "from_prophet_cv",
    "history_at",
    "monthly_cutoffs",
    "months_between",
    "prophet_cross_validation",
    "rolling_origin_backtest",
]

VINTAGE_COLUMN: Final = "vintage"
_log = get_logger(__name__)


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole calendar months from ``start`` to ``end``, element-wise."""
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def history_at(frame: SeriesFrame, cutoff: pd.Timestamp) -> SeriesFrame:
    """Rows of ``frame`` that were known at ``cutoff`` (``ds`` and vintage <= cutoff)."""
    mask = frame["ds"] <= cutoff
    if VINTAGE_COLUMN in frame.columns:
        mask &= pd.to_datetime(frame[VINTAGE_COLUMN]) <= cutoff
    return frame.loc[mask].sort_values("ds").reset_index(drop=True)


def monthly_cutoffs(
    frame: SeriesFrame,
    *,
    initial: int = 60,
    period: int = 1,
    min_horizon: int = 1,
) -> list[pd.Timestamp]:
    """Monthly cutoffs for one series.

    The first cutoff leaves ``initial`` months of training data; later ones step
    by ``period`` months; the last leaves at least ``min_horizon`` months of
    actuals after it.
    """
    if initial < 1 or period < 1 or min_horizon < 1:
        raise ValueError("initial, period and min_horizon must be >= 1")
    ds = pd.to_datetime(frame["ds"]).sort_values()
    first = ds.iloc[0] + pd.DateOffset(months=initial - 1)
    last = ds.iloc[-1] - pd.DateOffset(months=min_horizon)
    if first > last:
        return []
    return list(pd.date_range(first, last, freq=f"{period}MS"))


@dataclass(frozen=True, slots=True)
class _Task:
    forecaster: Forecaster
    history: pd.DataFrame
    series_id: str
    cutoff: pd.Timestamp
    horizon: int
    use_cache: bool


@cached_fit(version="1")
def _cached_fit_predict(forecaster: Forecaster, history: pd.DataFrame, horizon: int) -> Any:
    return forecaster.fit_predict(history, horizon)


def _run_task(task: _Task) -> pd.DataFrame:
    last_seen = pd.Timestamp(task.history["ds"].iloc[-1])
    # With publication lags the newest visible month can precede the cutoff; forecast
    # through the gap and keep only targets after the cutoff.
    gap = (task.cutoff.year - last_seen.year) * 12 + (task.cutoff.month - last_seen.month)
    steps = task.horizon + gap
    if task.use_cache:
        fc = _cached_fit_predict(task.forecaster, task.history, steps)
    else:
        fc = task.forecaster.fit_predict(task.history, steps)
    fc = validate_forecast_frame(pd.DataFrame(fc).reset_index(drop=True), steps)
    if (fc["ds"] <= last_seen).any():
        raise ValueError(
            f"forecaster returned a target inside its history (cutoff {task.cutoff:%Y-%m})"
        )
    fc = fc.loc[fc["ds"] > task.cutoff].copy()
    fc.insert(0, "series_id", task.series_id)
    fc.insert(1, "cutoff", task.cutoff)
    fc["horizon"] = months_between(fc["cutoff"], fc["ds"]).astype("int64")
    return fc


def _attach_actuals(forecasts: pd.DataFrame, frame: SeriesFrame) -> BacktestFrame:
    # Realised values use the latest vintage: the truth is what we know now.
    actuals = frame[["series_id", "ds", "y"]].drop_duplicates(["series_id", "ds"], keep="last")
    merged = forecasts.merge(actuals, on=["series_id", "ds"], how="inner")
    extra = list(quantile_columns(merged).values())
    cols = [*BACKTEST_COLUMNS, *extra]
    out = merged[cols].sort_values(["series_id", "cutoff", "horizon"]).reset_index(drop=True)
    return validate_backtest_frame(out)


def rolling_origin_backtest(
    frame: SeriesFrame,
    forecaster: Forecaster,
    horizon: int,
    *,
    cutoffs: Sequence[pd.Timestamp] | None = None,
    initial: int = 60,
    period: int = 1,
    min_history: int = 24,
    n_jobs: int | None = None,
    use_cache: bool = True,
) -> BacktestFrame:
    """Refit ``forecaster`` at every cutoff of every series and score against actuals.

    Args:
        frame: ``SeriesFrame`` with one or more series, optionally with a
            ``vintage`` column.
        forecaster: Model satisfying :class:`~kairos.vinge.schema.Forecaster`.
        horizon: Months forecast after each cutoff.
        cutoffs: Explicit cutoffs (applied to every series). Default: per-series
            :func:`monthly_cutoffs` with ``initial`` and ``period``.
        initial: Training months before the first default cutoff.
        period: Months between default cutoffs.
        min_history: Skip a cutoff whose visible history has fewer rows.
        n_jobs: Workers for :func:`kairos.compute.pmap` (``None`` uses settings).
        use_cache: Memoise each fit on disk with :func:`kairos.compute.cached_fit`.

    Returns:
        A ``BacktestFrame``. Targets without a realised value are dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    tasks: list[_Task] = []
    for series_id, group in frame.groupby("series_id", sort=True):
        group = group.sort_values("ds")
        series_cutoffs = (
            list(cutoffs)
            if cutoffs is not None
            else monthly_cutoffs(group, initial=initial, period=period)
        )
        for cutoff in series_cutoffs:
            history = history_at(group, pd.Timestamp(cutoff))
            if len(history) < min_history:
                continue
            tasks.append(
                _Task(forecaster, history, str(series_id), pd.Timestamp(cutoff), horizon, use_cache)
            )
    _log.info("backtest: %d fits over %d series", len(tasks), frame["series_id"].nunique())
    if not tasks:
        return pd.DataFrame(columns=list(BACKTEST_COLUMNS))
    forecasts = pd.concat(pmap(_run_task, tasks, n_jobs=n_jobs), ignore_index=True)
    return _attach_actuals(forecasts, frame)


def from_prophet_cv(
    cv: pd.DataFrame,
    series_id: str,
    *,
    log_space: bool = True,
    max_horizon: int | None = None,
) -> BacktestFrame:
    """Convert ``prophet.diagnostics.cross_validation`` output to a ``BacktestFrame``.

    Args:
        cv: Frame with ``ds, yhat, yhat_lower, yhat_upper, y, cutoff``.
        series_id: Identifier to stamp on every row.
        log_space: The model was fitted on ``ln y`` (KAIROS null models are), so
            exponentiate back to raw units.
        max_horizon: Drop rows further than this many months after the cutoff.
    """
    out = cv.rename(columns={"yhat_lower": "yhat_lo", "yhat_upper": "yhat_hi"}).copy()
    for col in ("ds", "cutoff"):
        out[col] = pd.to_datetime(out[col]).dt.to_period("M").dt.to_timestamp()
    if log_space:
        for col in ("y", "yhat", "yhat_lo", "yhat_hi"):
            out[col] = np.exp(out[col].astype("float64"))
    out["series_id"] = series_id
    out["horizon"] = months_between(out["cutoff"], out["ds"]).astype("int64")
    out = out.loc[out["horizon"] >= 1]
    if max_horizon is not None:
        out = out.loc[out["horizon"] <= max_horizon]
    out = out[list(BACKTEST_COLUMNS)].sort_values(["cutoff", "horizon"]).reset_index(drop=True)
    return validate_backtest_frame(out)


def prophet_cross_validation(
    model: Any,
    series_id: str,
    horizon: int,
    *,
    initial: int = 60,
    period: int = 1,
    log_space: bool = True,
    parallel: Literal["processes", "threads", "dask"] | None = "processes",
) -> BacktestFrame:
    """Run Prophet's own rolling-origin CV on monthly cutoffs.

    Args:
        model: A **fitted** ``prophet.Prophet`` instance.
        series_id: Identifier for the output rows.
        horizon: Months to forecast after each cutoff.
        initial: Training months before the first cutoff.
        period: Months between cutoffs.
        log_space: The model was fitted on ``ln y``.
        parallel: Passed to ``cross_validation`` (``"processes"`` on this
            workstation; ``None`` inside an already-parallel outer pool).
    """
    from prophet.diagnostics import cross_validation

    history = model.history[["ds"]].copy()
    cutoffs = monthly_cutoffs(history, initial=initial, period=period)
    if not cutoffs:
        raise ValueError("series too short for the requested initial window")
    cv = cross_validation(
        model,
        horizon=f"{31 * horizon + 1} days",
        cutoffs=cutoffs,
        parallel=parallel,
        disable_tqdm=True,
    )
    return from_prophet_cv(cv, series_id, log_space=log_space, max_horizon=horizon)
