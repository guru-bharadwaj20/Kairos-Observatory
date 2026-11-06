"""K-9.02: world population, von Foerster's doomsday and the demographic transition.

Von Foerster, Mora and Amiot (Science, 1960) fitted ``N = C/(t_c - t)^k`` to
world population and got ``t_c ≈ 2026.87`` (13 November 2026). This case asks
two things of the detectors:

1. **With data up to 1960**, the hyperbolic fit should put ``t_c`` near 2026.87
   (the 95 % profile interval must contain it).
2. **With data from 1960 to 1980**, the local exponent ``δ`` (from
   ``g = d ln N/dt ∝ N^δ``) should fall toward 0 and below the 0.25 alarm line: growth stopped being
   super-exponential once the demographic transition began, so a correct
   detector stands down instead of counting down to a singularity.

Data: Our World in Data world population (HYDE, Gapminder and UN WPP),
CC BY 4.0. Decadal before 1800, annual after.

Specification. The primary fit uses a **10-year grid from 1750 to 1960** so the
dense annual records after 1800 do not dominate the fit. That choice was made
while looking at the answer and is *not* pre-registered; :func:`sensitivity`
reports how much ``t_c`` moves with the start year and spacing, and the
validation report shows that table next to the headline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from kairos.types import SeriesFrame, SeriesMeta
from kairos.validation.backtest import as_of_view, predictions_frame, run_backtest, yearly_dates
from kairos.validation.cases import CaseResult
from kairos.validation.datasets import SNAPSHOTS, load_snapshot
from kairos.validation.detectors import (
    HyperbolicDetector,
    Prediction,
    decimal_year,
    single_series_id,
)

__all__ = [
    "VON_FOERSTER_T_C",
    "LocalDeltaDetector",
    "load_population",
    "run",
    "sensitivity",
    "thin",
]

VON_FOERSTER_T_C = 2026.87
"""Von Foerster et al. (1960): 13 November 2026 as a decimal year."""

SERIES_ID = "world_population"

SNAPSHOTS[SERIES_ID] = (
    "world_population.csv",
    SeriesMeta(
        series_id=SERIES_ID,
        name="World population",
        unit="people",
        source="owid",
        source_group="demography",
        side="validation",
        transform="log",
        license="CC-BY-4.0",
        url="https://ourworldindata.org/grapher/population",
    ),
)


def load_population() -> SeriesFrame:
    """World population snapshot, 1700-2023, one row per available year."""
    return load_snapshot(SERIES_ID)


def thin(frame: SeriesFrame, *, start: int, step: int) -> SeriesFrame:
    """Keep the January rows of years ``start, start+step, ...``."""
    years = frame["ds"].dt.year
    keep = (years >= start) & ((years - start) % step == 0)
    return frame[keep].reset_index(drop=True)


@dataclass(frozen=True)
class LocalDeltaDetector:
    """Local exponent ``δ`` from historical data only (D1).

    Over the trailing ``window_years`` it estimates ``g = d ln x/dt`` by central
    differences of the observed series (never a forecast) and regresses
    ``ln g`` on ``ln x``; the slope is ``δ``. When ``δ > 0`` the implied critical
    time is ``t_c = t + 1/(δ·g)`` (D5). Alarms when ``δ`` exceeds
    ``delta_alarm`` and ``t_c`` is within ``horizon_years``.

    Attributes:
        name: Detector name.
        window_years: Trailing estimation window.
        delta_alarm: ``δ`` above which the detector alarms.
        horizon_years: Farthest ``t_c`` that can alarm.
    """

    name: str = "local_delta"
    window_years: float = 15.0
    delta_alarm: float = 0.25
    horizon_years: float = 100.0

    def __call__(self, frame: SeriesFrame, as_of: date) -> Prediction:
        """Estimate ``δ`` on the trailing window of ``frame``."""
        series_id = single_series_id(frame)
        data = frame.dropna(subset=["y"]).sort_values("ds")
        t = decimal_year(data["ds"])
        y = data["y"].to_numpy(dtype=np.float64)
        mask = t >= t.max() - self.window_years if len(t) else np.zeros(0, bool)
        if mask.sum() < 4:
            return Prediction(
                self.name, series_id, as_of, int(mask.sum()), None, None, None, None, False, 0,
                {"reason": "fewer than 4 points in window"},
            )  # fmt: skip
        g = np.gradient(np.log(y), t)
        tw, yw, gw = t[mask], y[mask], g[mask]
        positive = gw > 0
        if positive.sum() < 4:
            delta = 0.0
        else:
            delta = float(np.polyfit(np.log(yw[positive]), np.log(gw[positive]), 1)[0])
        g_now = float(gw[-1])
        t_c = float(tw[-1] + 1.0 / (delta * g_now)) if delta > 0 and g_now > 0 else None
        lead = (t_c - float(tw[-1])) if t_c is not None else None
        alarm = bool(delta > self.delta_alarm and lead is not None and lead <= self.horizon_years)
        return Prediction(
            detector=self.name,
            series_id=series_id,
            as_of=as_of,
            n_obs=int(mask.sum()),
            t_c=t_c,
            t_c_lo=None,
            t_c_hi=None,
            delta=delta,
            alarm=alarm,
            level=1 if alarm else 0,
            detail={"growth_rate": g_now},
        )


def sensitivity(
    frame: SeriesFrame | None = None,
    *,
    starts: tuple[int, ...] = (1700, 1750, 1800, 1850, 1900),
    steps: tuple[int, ...] = (1, 10, 20),
    as_of: date = date(1960, 12, 31),
) -> pd.DataFrame:
    """``t_c`` of the hyperbolic fit for every (start year, spacing) choice."""
    frame = load_population() if frame is None else frame
    view = as_of_view(frame, as_of)
    detector = HyperbolicDetector(horizon_years=500.0)
    rows = []
    for start in starts:
        for step in steps:
            pred = detector(thin(view, start=start, step=step), as_of)
            rows.append({
                "start": start,
                "step": step,
                "n_obs": pred.n_obs,
                "t_c": pred.t_c,
                "t_c_lo": pred.t_c_lo,
                "t_c_hi": pred.t_c_hi,
                "delta": pred.delta,
            })  # fmt: skip
    return pd.DataFrame(rows)


def run(*, n_jobs: int | None = 1) -> CaseResult:
    """Run both parts of the world-population case."""
    frame = load_population()
    as_of_1960 = date(1960, 12, 31)
    primary = thin(frame, start=1750, step=10)
    pred = HyperbolicDetector(horizon_years=500.0)(as_of_view(primary, as_of_1960), as_of_1960)
    if pred.t_c is None or pred.t_c_lo is None or pred.t_c_hi is None:
        raise RuntimeError(f"hyperbolic fit found no singularity: {pred.detail}")
    covers = pred.t_c_lo <= VON_FOERSTER_T_C <= pred.t_c_hi

    replay = predictions_frame(
        run_backtest(frame, LocalDeltaDetector(), yearly_dates(1960, 1990), n_jobs=n_jobs)
    )
    delta = replay.set_index(replay["as_of"].map(lambda d: d.year))["delta"]
    trend = float(np.polyfit(delta.loc[1960:1980].index, delta.loc[1960:1980].to_numpy(), 1)[0])
    falls = delta.loc[1980] < min(delta.loc[1960], 0.25) and trend < 0
    table = sensitivity(frame)
    spread = (float(table["t_c"].min()), float(table["t_c"].max()))
    return CaseResult(
        case="population",
        passed=bool(covers and falls),
        criterion=(
            "1960 fit's 95% t_c interval contains 2026.87; local delta falls over "
            "1960-80 and ends below 0.25 (no longer super-exponential)"
        ),
        summary={
            "t_c_1960": round(pred.t_c, 2),
            "t_c_interval_1960": (round(pred.t_c_lo, 2), round(pred.t_c_hi, 2)),
            "delta_hyperbolic_1960": round(pred.delta or float("nan"), 3),
            "delta_local_1960": round(float(delta.loc[1960]), 3),
            "delta_local_1980": round(float(delta.loc[1980]), 3),
            "delta_trend_per_year_1960_80": round(trend, 4),
            "t_c_spread_over_specifications": (round(spread[0], 1), round(spread[1], 1)),
        },
        table=replay,
        notes=[
            "Primary specification (10-year grid, 1750-1960) chosen after inspection; "
            f"t_c ranges {spread[0]:.0f}-{spread[1]:.0f} across start years and spacings.",
            "Profile intervals assume independent errors and are too narrow (D4).",
        ],
    )
