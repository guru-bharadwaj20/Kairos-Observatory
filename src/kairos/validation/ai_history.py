"""K-9.07: AI history replay, 2012 to today, as of each year.

At every 1 January from 2012 the replay sees only models published before that
date, builds the frontier of training compute, and reports

* the **changepoints** in the log-growth rate (segmented regression with up to
  two breaks, chosen by BIC), and
* the **acceleration dial** from the hyperbolic baseline detector.

This is a descriptive replay, not a test with a known ending: it passes when
every year produces a reading without lookahead. Its value is the record of
what the dials would have said at the time.

Data: Epoch AI, "Data on Notable AI Models" (models from 2010 with a compute
estimate), CC BY 4.0. The frontier is the running maximum, carried forward
month by month; the dedicated envelope builder (K-2.06, D7) replaces it once
the data layer lands.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd

from kairos.types import SeriesFrame, validate_series_frame
from kairos.validation.backtest import as_of_view, yearly_dates
from kairos.validation.cases import CaseResult
from kairos.validation.datasets import read_snapshot_csv
from kairos.validation.detectors import Detector, HyperbolicDetector, decimal_year

__all__ = ["Changepoints", "frontier_frame", "load_models", "run", "segmented_changepoints"]

FloatArray = npt.NDArray[np.float64]
SERIES_ID = "frontier_training_compute"


def load_models() -> pd.DataFrame:
    """Per-model snapshot: ``publication_date``, ``model``, ``training_compute_flop``."""
    raw = read_snapshot_csv("epoch_notable_compute.csv")
    raw["publication_date"] = pd.to_datetime(raw["publication_date"])
    return raw


def frontier_frame(models: pd.DataFrame, as_of: date) -> SeriesFrame:
    """Monthly running-max frontier built from models published by ``as_of``."""
    seen = models[models["publication_date"] <= pd.Timestamp(as_of)]
    if seen.empty:
        return pd.DataFrame(
            {"ds": pd.Series(dtype="datetime64[ns]"), "y": pd.Series(dtype=float), "series_id": []}
        )
    month = seen["publication_date"].dt.to_period("M").dt.to_timestamp()
    best = seen.groupby(month)["training_compute_flop"].max()
    idx = pd.date_range(best.index.min(), best.index.max(), freq="MS")
    y = best.reindex(idx).ffill().cummax()
    frame = pd.DataFrame({
        "ds": idx.astype("datetime64[ns]"),
        "y": y.to_numpy(dtype=float),
        "series_id": SERIES_ID,
        "source": "epoch",
        "source_group": "epoch",
        "unit": "FLOP",
        "available_from": idx,
    })  # fmt: skip
    return validate_series_frame(frame)


@dataclass(frozen=True, slots=True)
class Changepoints:
    """Segmented log-linear fit: break times and per-segment doubling times."""

    breaks: tuple[float, ...]
    slopes: tuple[float, ...]
    bic: float

    @property
    def doubling_months(self) -> tuple[float, ...]:
        """Doubling time of each segment in months (``inf`` for flat segments)."""
        return tuple(12 * math.log(2) / s if s > 0 else math.inf for s in self.slopes)


def _hinge_fit(
    t: FloatArray, ly: FloatArray, breaks: tuple[float, ...]
) -> tuple[float, FloatArray]:
    cols = [np.ones_like(t), t] + [np.maximum(0.0, t - b) for b in breaks]
    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    sse = float(((ly - design @ coef) ** 2).sum())
    return sse, np.asarray(coef, dtype=np.float64)


def segmented_changepoints(
    t: FloatArray, y: FloatArray, *, max_breaks: int = 2, min_segment_years: float = 1.5
) -> Changepoints:
    """Continuous piecewise-linear fit of ``ln y`` with 0 to ``max_breaks`` breaks.

    Candidate breaks are a quarterly grid; segments shorter than
    ``min_segment_years`` are not allowed. The number of breaks minimises BIC.
    """
    ly = np.log(y)
    n = len(t)
    grid = np.arange(t.min() + min_segment_years, t.max() - min_segment_years, 0.25)
    best: Changepoints | None = None
    for k in range(max_breaks + 1):
        for combo in itertools.combinations(grid, k):
            if k == 2 and combo[1] - combo[0] < min_segment_years:
                continue
            sse, coef = _hinge_fit(t, ly, tuple(combo))
            bic = n * math.log(max(sse, 1e-300) / n) + (2 + 2 * k) * math.log(n)
            if best is None or bic < best.bic:
                slopes = tuple(float(s) for s in np.cumsum(coef[1:]))
                best = Changepoints(tuple(float(b) for b in combo), slopes, bic)
    if best is None:
        raise ValueError("not enough data for a changepoint fit")
    return best


def run(*, detector: Detector | None = None, last_year: int | None = None) -> CaseResult:
    """Replay the AI frontier as of 1 January of every year from 2012."""
    detector = detector or HyperbolicDetector()
    models = load_models()
    last_year = last_year or int(models["publication_date"].max().year)
    rows = []
    for as_of in yearly_dates(2012, last_year):
        frame = as_of_view(frontier_frame(models, as_of), as_of)
        pred = detector(frame, as_of)
        t = decimal_year(frame["ds"])
        cps = segmented_changepoints(t, frame["y"].to_numpy()) if len(t) >= 24 else None
        rows.append({
            "as_of": as_of,
            "n_months": len(frame),
            "frontier_flop": float(frame["y"].iloc[-1]),
            "breaks": tuple(round(b, 2) for b in cps.breaks) if cps else (),
            "doubling_months_last": round(cps.doubling_months[-1], 1) if cps else None,
            "t_c": pred.t_c,
            "delta": pred.delta,
            "alarm": pred.alarm,
            "level": pred.level,
        })  # fmt: skip
    table = pd.DataFrame(rows)
    complete = len(table) == last_year - 2011
    return CaseResult(
        case="ai_history",
        passed=complete,
        criterion="a dial reading and changepoint fit for every year, using only prior data",
        summary={
            "years": len(table),
            "alarm_years": [d.year for d in table.loc[table["alarm"], "as_of"]],
            "max_level": int(table["level"].max()),
            "latest_breaks": tuple(round(b, 2) for b in table["breaks"].iloc[-1]),
            "latest_doubling_months": table["doubling_months_last"].iloc[-1],
        },
        table=table,
        notes=[
            "Descriptive replay: there is no known ending to score against.",
            "Running-max frontier creates flat steps (D7); publication date is used as "
            "the data vintage, so later compute re-estimates are a mild lookahead.",
        ],
    )
