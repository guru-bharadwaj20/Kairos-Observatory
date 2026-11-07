"""K-9.03 / K-9.04: speculative bubbles with known crash dates.

Index levels and exchange prices are not openly licensed, so they are fetched
from FRED at run time (daily closes, cached locally, never committed). Without
network access and without a cache the cases report ``skipped``.

LPPLS is fitted on **daily** data, as in the published work; the month-start
rule of the ``SeriesFrame`` convention is relaxed for these frames only. Each
as-of date sees closes up to the previous day (one-day publication lag).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from kairos.types import SeriesFrame
from kairos.validation.backtest import predictions_frame, run_backtest
from kairos.validation.cases import CaseResult
from kairos.validation.datasets import DataUnavailableError, fetch_fred
from kairos.validation.detectors import Detector, decimal_year
from kairos.validation.lppls import LPPLSDetector

__all__ = ["DOTCOM", "BubbleCase", "monthly_dates", "run_bubble", "run_dotcom"]


@dataclass(frozen=True)
class BubbleCase:
    """A bubble with a known peak and a published or reference ``t_c`` window.

    Attributes:
        name: Case name.
        fred_id: FRED series code.
        series_id: KAIROS series id.
        unit: Unit of the prices.
        peak: Date of the all-time high that ended the bubble.
        reference_window: ``(lo, hi)`` decimal years a correct ``t_c`` should hit.
        reference: Where the window comes from.
        replay: ``(first, last)`` as-of months to replay.
    """

    name: str
    fred_id: str
    series_id: str
    unit: str
    peak: date
    reference_window: tuple[float, float]
    reference: str
    replay: tuple[date, date]


DOTCOM = BubbleCase(
    name="dotcom",
    fred_id="NASDAQCOM",
    series_id="nasdaq_composite",
    unit="index",
    peak=date(2000, 3, 10),
    reference_window=(2000.0, 2000.5),
    reference=(
        "Johansen & Sornette (2000), 'The Nasdaq crash of April 2000', arXiv:cond-mat/0004263: "
        "LPPLS critical time in the first half of 2000 (window used here: 2000.0-2000.5)"
    ),
    replay=(date(1999, 1, 1), date(2000, 12, 1)),
)


def monthly_dates(first: date, last: date) -> list[date]:
    """Month-start dates from ``first`` to ``last`` inclusive."""
    return [d.date() for d in pd.date_range(first, last, freq="MS")]


def run_bubble(
    case: BubbleCase,
    *,
    frame: SeriesFrame | None = None,
    detector: Detector | None = None,
    n_jobs: int | None = None,
) -> CaseResult:
    """Replay ``detector`` (default LPPLS) month by month through a bubble.

    Pass criterion: the last as-of date before the peak raises an alarm and its
    ``t_c`` lies inside ``case.reference_window``. Alarms after the peak are
    counted as false alarms and reported.
    """
    if frame is None:
        try:
            frame = fetch_fred(
                case.fred_id, series_id=case.series_id, unit=case.unit, frequency=None
            )
        except DataUnavailableError as exc:
            return CaseResult(case.name, False, {"status": "data unavailable"},
                              "needs network or a cached FRED download", notes=[str(exc)],
                              skipped=True)  # fmt: skip
    detector = detector or LPPLSDetector(min_points=100)
    dates = monthly_dates(*case.replay)
    table = predictions_frame(
        run_backtest(frame, detector, dates, publication_lag_days=1, n_jobs=n_jobs)
    )
    peak = pd.Timestamp(case.peak).date()
    before = table[table["as_of"] <= peak]
    after = table[table["as_of"] > peak]
    last = before.iloc[-1]
    lo, hi = case.reference_window
    hit = bool(last["alarm"] and last["t_c"] is not None and lo <= last["t_c"] <= hi)
    alarms = before[before["alarm"]]
    first_alarm = alarms["as_of"].iloc[0] if len(alarms) else None
    lead_days = (peak - first_alarm).days if first_alarm is not None else None
    t_c_last = None if pd.isna(last["t_c"]) else round(float(last["t_c"]), 3)
    return CaseResult(
        case=case.name,
        passed=hit,
        criterion=f"last pre-peak replay alarms with t_c in [{lo}, {hi}]",
        summary={
            "peak": peak.isoformat(),
            "peak_decimal_year": round(decimal_year(peak), 3),
            "last_pre_peak_as_of": last["as_of"].isoformat(),
            "t_c_last_pre_peak": t_c_last,
            "first_alarm": first_alarm.isoformat() if first_alarm else None,
            "lead_time_days": lead_days,
            "post_peak_alarm_rate": round(float(after["alarm"].mean()), 3) if len(after) else None,
        },
        table=table,
        notes=[
            case.reference,
            "Baseline LPPLS alarms persistently; post-peak alarms are false alarms "
            "of a detector with no crash-confirmation logic.",
        ],
    )


def run_dotcom(*, n_jobs: int | None = None) -> CaseResult:
    """K-9.03: Nasdaq Composite 1999-2000."""
    return run_bubble(DOTCOM, n_jobs=n_jobs)
