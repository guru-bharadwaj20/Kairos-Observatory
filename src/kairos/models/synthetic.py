"""Synthetic series with known structure, for tests, examples and benchmarks.

Every generator returns a valid :data:`kairos.types.SeriesFrame` in raw units.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from kairos.types import SeriesFrame, validate_series_frame

__all__ = ["piecewise_exponential", "synthetic_ai_compute"]


def piecewise_exponential(
    start: str = "2000-01-01",
    end: str = "2026-01-01",
    *,
    breaks: Sequence[str] = (),
    rates: Sequence[float] = (0.5,),
    level: float = 1.0,
    noise: float = 0.0,
    seed: int = 0,
    series_id: str = "synthetic",
    unit: str = "index",
) -> SeriesFrame:
    """Monthly series whose log grows piecewise linearly.

    Args:
        start: First month.
        end: Last month (inclusive).
        breaks: Dates where the growth rate changes; ``len(rates) == len(breaks) + 1``.
        rates: Growth rates in natural-log units **per year** for each regime.
        level: Value at ``start``.
        noise: Standard deviation of i.i.d. Gaussian noise on ``ln y``.
        seed: NumPy seed for the noise.
        series_id: ``series_id`` column value.
        unit: ``unit`` column value.
    """
    if len(rates) != len(breaks) + 1:
        raise ValueError("need exactly one more rate than breaks")
    ds = pd.date_range(start, end, freq="MS")
    years = ((ds - ds[0]).days / 365.25).to_numpy(dtype="float64")
    edges = [(pd.Timestamp(b) - ds[0]).days / 365.25 for b in breaks]
    slope = np.full(len(ds), rates[0], dtype="float64")
    for edge, rate in zip(edges, rates[1:], strict=True):
        slope[years >= edge] = rate
    dt = np.diff(years, prepend=years[0])
    log_y = np.log(level) + np.cumsum(slope * dt)
    if noise > 0:
        log_y = log_y + np.random.default_rng(seed).normal(0.0, noise, len(ds))
    frame = pd.DataFrame(
        {
            "ds": ds.astype("datetime64[ns]"),
            "y": np.exp(log_y),
            "series_id": series_id,
            "unit": unit,
        }
    )
    return validate_series_frame(frame)


def synthetic_ai_compute(*, noise: float = 0.15, seed: int = 0) -> SeriesFrame:
    """Stylised frontier-compute history 2000-2026 with shifts in 2012, 2017 and 2021.

    Growth rates (ln units per year) step from 0.35 (Moore era) to 1.2, 1.8 and 2.5.
    Used to test that changepoint detection recovers known shifts blind.
    """
    return piecewise_exponential(
        "2000-01-01",
        "2026-06-01",
        breaks=("2012-06-01", "2017-06-01", "2021-06-01"),
        rates=(0.35, 1.2, 1.8, 2.5),
        noise=noise,
        seed=seed,
        series_id="ai_compute_frontier",
        unit="FLOP",
    )
