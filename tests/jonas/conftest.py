"""Synthetic C/H SeriesFrames for the Jonas Gap tests (no network, no data layer)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import pytest

FrameFactory = Callable[..., pd.DataFrame]


def make_frame(
    c_rates: Mapping[str, float] | None = None,
    h_rates: Mapping[str, float] | None = None,
    *,
    start: str = "2015-01-01",
    periods: int = 120,
    noise: float = 0.0,
    seed: int = 0,
    band: float | None = None,
) -> pd.DataFrame:
    """Build exponential series ``y = level * exp(rate * years + noise)``.

    Rates are continuous growth per year in log units.
    """
    c_rates = {"c_compute": 0.9, "c_bench": 0.7} if c_rates is None else c_rates
    h_rates = {"h_papers": 0.3, "h_funding": 0.2} if h_rates is None else h_rates
    rng = np.random.default_rng(seed)
    ds = pd.date_range(start, periods=periods, freq="MS")
    years = np.arange(periods) / 12.0
    rows = []
    for group, rates in (("C", c_rates), ("H", h_rates)):
        for i, (sid, rate) in enumerate(rates.items()):
            log_y = np.log(10.0 + 5 * i) + rate * years + noise * rng.standard_normal(periods)
            part = pd.DataFrame(
                {
                    "ds": ds,
                    "y": np.exp(log_y),
                    "series_id": sid,
                    "source": "synthetic",
                    "source_group": group,
                    "unit": "index",
                }
            )
            if band is not None:
                part["y_lo"] = part["y"] * np.exp(-band)
                part["y_hi"] = part["y"] * np.exp(band)
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def frame_factory() -> FrameFactory:
    return make_frame
