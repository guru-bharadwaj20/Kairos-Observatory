"""Reference forecaster for the harness: a random walk with drift in ``ln y``.

It satisfies :class:`~kairos.vinge.schema.Forecaster`, needs no Stan or torch,
and gives the floor any real model should beat. Its band grows like
``sqrt(h)``, so on a pure log-linear trend its ``h*`` is set by the noise level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from kairos.types import SeriesFrame
from kairos.vinge.schema import DEFAULT_INTERVAL, ForecastFrame

__all__ = ["LogDriftForecaster"]


@dataclass(frozen=True, slots=True)
class LogDriftForecaster:
    """Random walk with drift on ``ln y``; Gaussian bands on the log scale.

    Attributes:
        interval_width: Coverage of ``yhat_lo``/``yhat_hi``.
        window: Use only the last ``window`` months to estimate drift and noise
            (``None`` uses all history).
    """

    interval_width: float = DEFAULT_INTERVAL
    window: int | None = None

    def fit_predict(self, history: SeriesFrame, horizon: int) -> ForecastFrame:
        """Forecast ``horizon`` months after the last row of ``history``."""
        hist = history.sort_values("ds")
        if self.window is not None:
            hist = hist.tail(self.window)
        logy = np.log(hist["y"].to_numpy(dtype="float64"))
        if logy.size < 3:
            raise ValueError("need at least 3 points")
        steps = np.diff(logy)
        drift = float(steps.mean())
        sigma = float(steps.std(ddof=1)) or 1e-12
        h = np.arange(1, horizon + 1, dtype="float64")
        centre = logy[-1] + drift * h
        z = float(norm.ppf(0.5 + self.interval_width / 2))
        half = z * sigma * np.sqrt(h)
        ds = pd.date_range(
            pd.Timestamp(hist["ds"].iloc[-1]) + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )
        return pd.DataFrame(
            {
                "ds": ds,
                "yhat": np.exp(centre),
                "yhat_lo": np.exp(centre - half),
                "yhat_hi": np.exp(centre + half),
            }
        )
