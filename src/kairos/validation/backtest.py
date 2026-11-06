"""Backtest runner: replay a detector at past as-of dates with no lookahead.

For every as-of date the runner builds the **as-of view** of the data, the rows
that had been published by that date, and hands only that view to the detector.
Two rules decide what was published:

* ``ds <= as_of - publication_lag`` (a monthly figure is not known on its own
  month start), and
* if the frame has an ``available_from`` column (the data vintage, e.g. the date
  a revision was released), ``available_from <= as_of``. When several vintages
  of the same ``(series_id, ds)`` are present, the latest one visible at
  ``as_of`` wins.

Each as-of date is an independent task fanned out with :func:`kairos.compute.pmap`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from kairos.compute import pmap
from kairos.logging import get_logger
from kairos.types import SeriesFrame
from kairos.validation.detectors import Detector, Prediction

__all__ = ["BacktestTask", "as_of_view", "predictions_frame", "run_backtest", "yearly_dates"]

log = get_logger(__name__)

VINTAGE_COLUMN = "available_from"


def as_of_view(frame: SeriesFrame, as_of: date, *, publication_lag_days: int = 0) -> SeriesFrame:
    """Return the rows of ``frame`` that were available at ``as_of``.

    Args:
        frame: Long-format ``SeriesFrame``, optionally with ``available_from``.
        as_of: The replay date.
        publication_lag_days: Observations dated within this many days before
            ``as_of`` are treated as not yet published.
    """
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=publication_lag_days)
    view = frame[frame["ds"] <= cutoff]
    if VINTAGE_COLUMN in view.columns:
        view = view[pd.to_datetime(view[VINTAGE_COLUMN]) <= pd.Timestamp(as_of)]
        view = view.sort_values(VINTAGE_COLUMN).drop_duplicates(["series_id", "ds"], keep="last")
    return view.sort_values(["series_id", "ds"]).reset_index(drop=True)


@dataclass(frozen=True)
class BacktestTask:
    """One unit of work: run ``detector`` on ``frame`` as of ``as_of``."""

    frame: SeriesFrame
    detector: Detector
    as_of: date
    publication_lag_days: int = 0


def _run_task(task: BacktestTask) -> Prediction:
    view = as_of_view(task.frame, task.as_of, publication_lag_days=task.publication_lag_days)
    return task.detector(view, task.as_of)


def run_backtest(
    frame: SeriesFrame,
    detector: Detector,
    as_of_dates: Iterable[date],
    *,
    publication_lag_days: int = 0,
    n_jobs: int | None = 1,
) -> list[Prediction]:
    """Replay ``detector`` on one series at every date in ``as_of_dates``.

    Args:
        frame: One series in ``SeriesFrame`` form.
        detector: Any :class:`~kairos.validation.detectors.Detector`.
        as_of_dates: Replay dates; the output keeps their order.
        publication_lag_days: See :func:`as_of_view`.
        n_jobs: Worker processes (``None`` = ``Settings.n_jobs``; ``1`` in-process).

    Returns:
        One :class:`Prediction` per as-of date.
    """
    if len(frame["series_id"].unique()) != 1:
        raise ValueError("run_backtest expects exactly one series; loop over series_id")
    dates = list(as_of_dates)
    log.info(
        "backtest %s on %s: %d as-of dates", detector.name, frame["series_id"].iloc[0], len(dates)
    )
    tasks = [BacktestTask(frame, detector, d, publication_lag_days) for d in dates]
    return pmap(_run_task, tasks, n_jobs=n_jobs)


def predictions_frame(predictions: Sequence[Prediction]) -> pd.DataFrame:
    """Tabulate predictions, one row each."""
    return pd.DataFrame([p.as_record() for p in predictions])


def yearly_dates(start: int, end: int, *, month: int = 1, day: int = 1) -> list[date]:
    """As-of dates on the same day of every year from ``start`` to ``end`` inclusive."""
    return [date(year, month, day) for year in range(start, end + 1)]
