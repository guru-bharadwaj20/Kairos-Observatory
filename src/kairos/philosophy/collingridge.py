"""Collingridge lead time: how many months remain before a threshold is crossed.

Given ``n_paths`` simulated forecast paths over ``horizon`` monthly steps (step ``1``
is the first month after the data vintage), the lead time is the first month in
which the ``quantile``-th quantile across paths crosses the threshold. Acting on the
upper quantile follows Jonas: we plan for the pessimistic edge, not the median.

The interval is a percentile bootstrap over paths: resample paths with replacement,
recompute the crossing month, take the central ``ci`` interval. Resamples that never
cross within the horizon are *censored*; if the upper end is censored the interval is
open (``hi=None``, printed as ``>H``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from kairos.settings import get_settings

__all__ = ["LeadTime", "first_crossing", "lead_time"]

Direction = Literal["above", "below"]


class LeadTime(BaseModel):
    """Lead time with its bootstrap interval.

    Attributes:
        months: Point estimate, or ``None`` if not crossed within ``horizon``.
        lo: Lower end of the interval (``None`` when every resample is censored).
        hi: Upper end, or ``None`` if beyond the horizon.
        ci: Interval level.
        quantile: Forecast quantile that was tested against the threshold.
        horizon: Months simulated.
        censored_share: Share of bootstrap resamples that never crossed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    months: int | None
    lo: float | None
    hi: float | None
    ci: float = Field(gt=0, lt=1)
    quantile: float = Field(gt=0, lt=1)
    horizon: int = Field(ge=1)
    censored_share: float = Field(ge=0, le=1)

    def __str__(self) -> str:
        """E.g. ``"31 months (90% CI: 18–52)"``."""

        def fmt(x: float | None) -> str:
            return f">{self.horizon}" if x is None else f"{x:.0f}"

        pct = f"{self.ci * 100:.0f}%"
        if self.months is None:
            return f">{self.horizon} months (not crossed within the horizon)"
        lo = "?" if self.lo is None else fmt(self.lo)
        return f"{self.months} months ({pct} CI: {lo}–{fmt(self.hi)})"


def _as_paths(paths: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(paths, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("paths must be a 2-D array (n_paths >= 2, horizon >= 1)")
    if not np.isfinite(x).all():
        raise ValueError("paths must be finite")
    return x


def first_crossing(
    paths: ArrayLike, threshold: float, *, quantile: float, direction: Direction = "above"
) -> int | None:
    """First month (1-based) at which the quantile path crosses ``threshold``."""
    x = _as_paths(paths)
    q = quantile if direction == "above" else 1.0 - quantile
    band = np.quantile(x, q, axis=0)
    hit = band >= threshold if direction == "above" else band <= threshold
    idx = np.flatnonzero(hit)
    return int(idx[0]) + 1 if idx.size else None


def lead_time(
    paths: ArrayLike,
    threshold: float,
    *,
    quantile: float = 0.95,
    ci: float = 0.90,
    n_boot: int = 1000,
    direction: Direction = "above",
    seed: int | None = None,
) -> LeadTime:
    """Lead time with a percentile-bootstrap interval.

    Args:
        paths: ``(n_paths, horizon)`` forecast draws, monthly.
        threshold: Level whose crossing ends the lead time.
        quantile: Upper quantile tested (lower ``1 - quantile`` if ``direction="below"``).
        ci: Interval level.
        n_boot: Bootstrap resamples.
        direction: Crossing upwards (``"above"``) or downwards.
        seed: RNG seed (default: ``Settings.seed``).
    """
    x = _as_paths(paths)
    n, horizon = x.shape
    point = first_crossing(x, threshold, quantile=quantile, direction=direction)
    rng = np.random.default_rng(get_settings().seed if seed is None else seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        t = first_crossing(
            x[rng.integers(0, n, n)], threshold, quantile=quantile, direction=direction
        )
        boots[b] = np.inf if t is None else t
    censored = float(np.isinf(boots).mean())
    alpha = (1.0 - ci) / 2
    lo_q, hi_q = np.quantile(boots, [alpha, 1.0 - alpha], method="inverted_cdf")
    return LeadTime(
        months=point,
        lo=None if np.isinf(lo_q) else float(lo_q),
        hi=None if np.isinf(hi_q) else float(hi_q),
        ci=ci,
        quantile=quantile,
        horizon=horizon,
        censored_share=censored,
    )
