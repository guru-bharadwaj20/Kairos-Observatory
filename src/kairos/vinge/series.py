"""``h*(t)``: the Vinge Horizon recomputed at every monthly cutoff (K-6.04).

The expensive part is the backtest: one fit per ``(series, cutoff)``, run once
by :func:`kairos.vinge.backtest.rolling_origin_backtest` and memoised with the
fit cache, so a monthly rerun only fits the newest cutoff. ``h*(t)`` is then
cheap: at evaluation month ``t`` it uses only forecasts whose cutoff lies in
the trailing window ``(t - window, t]`` and whose target ``ds <= t``, so every
reading uses information available at ``t`` and nothing later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Final, Literal

import pandas as pd

from kairos.compute import pmap
from kairos.logging import get_logger
from kairos.types import SeriesFrame
from kairos.vinge.backtest import rolling_origin_backtest
from kairos.vinge.horizon import band_horizon, error_horizon
from kairos.vinge.metrics import score_backtest
from kairos.vinge.schema import DEFAULT_INTERVAL, BacktestFrame, Forecaster

__all__ = ["HSTAR_COLUMNS", "HorizonSpec", "compute_hstar_series", "hstar_series"]

HSTAR_COLUMNS: Final = (
    "series_id",
    "as_of",
    "h_star",
    "censored",
    "crossed_at",
    "band_h_star",
    "band_censored",
    "n_cutoffs",
    "max_horizon",
)
_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HorizonSpec:
    """Parameters of one ``h*(t)`` run.

    Attributes:
        tolerance_factor: Error tolerance as a factor (``2.0`` = within 2x).
        metric: ``"ale"`` or ``"crps"``.
        k_efolds: Band-width threshold for the band variant.
        window: Trailing window of cutoffs, in months.
        min_cutoffs: Skip an evaluation month with fewer cutoffs in the window.
        smooth: Smoothing window over horizon (odd).
        interval_width: Nominal coverage of the forecast band.
    """

    tolerance_factor: float = 2.0
    metric: Literal["ale", "crps"] = "ale"
    k_efolds: float = 2.0
    window: int = 36
    min_cutoffs: int = 6
    smooth: int = 3
    interval_width: float = DEFAULT_INTERVAL


def _one_series(
    job: tuple[pd.DataFrame, tuple[pd.Timestamp, ...] | None, HorizonSpec],
) -> list[dict[str, object]]:
    scored, eval_dates, spec = job
    series_id = str(scored["series_id"].iloc[0])
    dates = eval_dates or tuple(pd.Timestamp(d) for d in sorted(scored["ds"].unique()))
    rows: list[dict[str, object]] = []
    for t in dates:
        lo = t - pd.DateOffset(months=spec.window)
        sub = scored.loc[(scored["cutoff"] > lo) & (scored["cutoff"] <= t) & (scored["ds"] <= t)]
        n_cut = int(sub["cutoff"].nunique())
        if n_cut < spec.min_cutoffs:
            continue
        err = error_horizon(sub, spec.tolerance_factor, metric=spec.metric, window=spec.smooth)
        band = band_horizon(sub, spec.k_efolds, window=spec.smooth)
        rows.append(
            {
                "series_id": series_id,
                "as_of": t,
                "h_star": err.h_star,
                "censored": err.censored,
                "crossed_at": err.crossed_at,
                "band_h_star": band.h_star,
                "band_censored": band.censored,
                "n_cutoffs": n_cut,
                "max_horizon": err.max_horizon,
            }
        )
    return rows


def hstar_series(
    bt: BacktestFrame,
    spec: HorizonSpec | None = None,
    *,
    eval_dates: Sequence[pd.Timestamp] | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Compute ``h*(t)`` for every series from a backtest.

    Args:
        bt: Full ``BacktestFrame`` (all cutoffs, all series).
        spec: Tolerance, window and smoothing; defaults to :class:`HorizonSpec`.
        eval_dates: Months ``t`` to evaluate; default every target month present.
        n_jobs: Workers (one task per series).

    Returns:
        One row per ``(series_id, as_of)`` with columns :data:`HSTAR_COLUMNS`.
    """
    spec = spec or HorizonSpec()
    scored = score_backtest(bt, spec.interval_width)
    dates = tuple(pd.Timestamp(d) for d in eval_dates) if eval_dates is not None else None
    jobs = [(group, dates, spec) for _, group in scored.groupby("series_id", sort=True)]
    rows = [row for part in pmap(_one_series, jobs, n_jobs=n_jobs) for row in part]
    out = pd.DataFrame(rows, columns=list(HSTAR_COLUMNS))
    out["crossed_at"] = out["crossed_at"].astype("Int64")
    _log.info("h*(t): %d readings, spec=%s", len(out), asdict(spec))
    return out.sort_values(["series_id", "as_of"]).reset_index(drop=True)


def compute_hstar_series(
    frame: SeriesFrame,
    forecaster: Forecaster,
    horizon: int,
    spec: HorizonSpec | None = None,
    *,
    initial: int = 60,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Backtest ``forecaster`` at every monthly cutoff, then compute ``h*(t)``.

    This is the largest CPU job in KAIROS; the per-cutoff fits are cached, so
    reruns only fit cutoffs that are new.
    """
    spec = spec or HorizonSpec(interval_width=forecaster.interval_width)
    bt = rolling_origin_backtest(frame, forecaster, horizon, initial=initial, n_jobs=n_jobs)
    return hstar_series(bt, spec, n_jobs=n_jobs)
