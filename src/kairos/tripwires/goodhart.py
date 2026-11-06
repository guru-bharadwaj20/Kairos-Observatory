"""Goodhart divergence detector.

When a measure becomes a target it stops being a good measure. KAIROS keeps
held-out metrics next to public ones; if the public metric's trend runs ahead of its
held-out twin, the gap is a gaming signal in its own right.

Both series are ``SeriesFrame`` objects in raw positive units. On the months they
share, each is fitted with a log-linear trend ``log y = a + b·t`` (``t`` in years)
using OLS with Newey-West (HAC) standard errors, because monthly residuals are
autocorrelated. The test is one-sided:

    ``z = (b_public − b_heldout) / sqrt(se_public² + se_heldout²)``

and the pair is flagged when ``p < alpha`` *and* the gap is at least ``min_gap``
log-units per year (``policy.yaml``, section ``goodhart``).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from kairos.logging import get_logger
from kairos.types import DialReading, SeriesFrame, validate_series_frame

__all__ = ["GoodhartConfig", "GoodhartResult", "detect_divergence", "log_trend"]

log = get_logger(__name__)


class GoodhartConfig(BaseModel):
    """Detector parameters (from ``policy.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = Field(default=0.05, gt=0, lt=1)
    min_gap: float = Field(default=0.0, ge=0)
    min_overlap: int = Field(default=12, ge=3)


class GoodhartResult(BaseModel):
    """Outcome of one public/held-out comparison.

    Attributes:
        public_slope: Log-trend of the public metric per year.
        heldout_slope: Log-trend of the held-out metric per year.
        gap: ``public_slope − heldout_slope``.
        se: Standard error of ``gap``.
        z: Test statistic.
        p_value: One-sided p-value of ``gap > 0``.
        n: Months used.
        flagged: Whether the gap counts as a gaming signal.
        as_of: Last shared month.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    public_slope: float
    heldout_slope: float
    gap: float
    se: float
    z: float
    p_value: float
    n: int
    flagged: bool
    as_of: date

    def to_reading(self, ci: float = 0.90) -> DialReading:
        """Express the result as a ``"goodhart"`` :class:`DialReading` (level 0 or 1)."""
        half = float(stats.norm.ppf(0.5 + ci / 2)) * self.se
        return DialReading(
            dial="goodhart",
            value=self.gap,
            level=1 if self.flagged else 0,
            ci_lo=self.gap - half,
            ci_hi=self.gap + half,
            as_of=self.as_of,
            detail=self.model_dump(mode="json"),
        )


def log_trend(t_years: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """Slope of ``log y`` on ``t`` and its Newey-West standard error."""
    n = len(y)
    lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    fit = sm.OLS(np.log(y), sm.add_constant(t_years)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return float(fit.params[1]), float(fit.bse[1])


def detect_divergence(
    public: SeriesFrame, heldout: SeriesFrame, config: GoodhartConfig | None = None
) -> GoodhartResult:
    """Compare the log-trends of a public metric and its held-out twin.

    Raises:
        ValueError: If the frames are invalid or share fewer than ``min_overlap`` months.
    """
    cfg = config or GoodhartConfig()
    pub = validate_series_frame(public)[["ds", "y"]].dropna()
    held = validate_series_frame(heldout)[["ds", "y"]].dropna()
    both = pub.merge(held, on="ds", suffixes=("_pub", "_held")).sort_values("ds")
    if len(both) < cfg.min_overlap:
        raise ValueError(f"only {len(both)} shared months, need {cfg.min_overlap}")
    t = ((both["ds"] - both["ds"].iloc[0]) / pd.Timedelta(days=365.25)).to_numpy(float)
    b_pub, se_pub = log_trend(t, both["y_pub"].to_numpy(float))
    b_held, se_held = log_trend(t, both["y_held"].to_numpy(float))
    gap = b_pub - b_held
    se = float(np.hypot(se_pub, se_held))
    z = gap / se if se > 0 else (np.inf if gap > 0 else 0.0)
    p = float(stats.norm.sf(z))
    flagged = bool(p < cfg.alpha and gap >= cfg.min_gap)
    if flagged:
        log.warning("Goodhart divergence: gap %.3f/yr, p=%.3g", gap, p)
    return GoodhartResult(
        public_slope=b_pub,
        heldout_slope=b_held,
        gap=gap,
        se=se,
        z=float(z),
        p_value=p,
        n=len(both),
        flagged=flagged,
        as_of=both["ds"].iloc[-1].date(),
    )
