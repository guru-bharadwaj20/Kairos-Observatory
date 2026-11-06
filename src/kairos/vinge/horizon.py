"""The ``h*`` definition (K-6.03, design note D3).

The idea sketch took ``ok["horizon"].max()``, the longest horizon that passes.
Errors are not monotone in horizon, so one lucky long horizon inflated ``h*``.
Here ``h*`` is the horizon just **before the first** horizon where the
*smoothed* error curve crosses the tolerance:

* :func:`error_horizon`: smoothed log-unit error (``ale`` or ``crps``) versus
  a tolerance stated as a factor, e.g. ``2.0`` = "within 2x" = ``ln 2`` e-folds;
* :func:`band_horizon`: the variant where the median width of the central
  (90 %) band exceeds ``k`` e-folds. It needs no actuals beyond the cutoff, so
  it can be read at the live edge.

If the curve never crosses, ``h*`` is the longest horizon tested and the result
is flagged ``censored`` (a lower bound, not a measurement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from kairos.vinge.metrics import error_curve

__all__ = [
    "HorizonResult",
    "band_horizon",
    "error_horizon",
    "first_crossing",
    "smooth_curve",
]


@dataclass(frozen=True, slots=True)
class HorizonResult:
    """One ``h*`` measurement.

    Attributes:
        h_star: Longest horizon (months) before the first crossing; ``0`` if the
            one-month forecast already fails.
        censored: The curve never crossed, so ``h_star`` is a lower bound.
        crossed_at: First failing horizon, or ``None`` when censored.
        metric: Curve the threshold was applied to.
        threshold: Threshold in e-folds.
        max_horizon: Longest horizon in the curve.
    """

    h_star: int
    censored: bool
    crossed_at: int | None
    metric: str
    threshold: float
    max_horizon: int


def smooth_curve(curve: pd.Series, window: int = 3) -> pd.Series:
    """Centred moving average over horizon, after filling gaps by interpolation.

    Args:
        curve: Error indexed by integer horizon.
        window: Odd window length in months; ``1`` disables smoothing.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    curve = curve.sort_index()
    full = curve.reindex(range(int(curve.index.min()), int(curve.index.max()) + 1))
    full = full.interpolate(limit_direction="both")
    return full.rolling(window, center=True, min_periods=1).mean()


def first_crossing(
    curve: pd.Series, threshold: float, *, metric: str, window: int = 3
) -> HorizonResult:
    """Apply the D3 rule to an error curve indexed by horizon (e-folds)."""
    if curve.empty:
        raise ValueError("empty error curve")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive number of e-folds")
    smoothed = smooth_curve(curve, window)
    over = smoothed.index[smoothed.to_numpy() > threshold]
    max_h = int(smoothed.index.max())
    if len(over) == 0:
        return HorizonResult(max_h, True, None, metric, threshold, max_h)
    crossed = int(over[0])
    return HorizonResult(crossed - 1, False, crossed, metric, threshold, max_h)


def error_horizon(
    scored: pd.DataFrame,
    tolerance_factor: float = 2.0,
    *,
    metric: Literal["ale", "crps"] = "ale",
    agg: Literal["median", "mean"] = "median",
    window: int = 3,
) -> HorizonResult:
    """``h*`` from forecast error: first horizon the smoothed error exceeds ``ln(factor)``.

    Args:
        scored: Output of :func:`kairos.vinge.metrics.score_backtest` (one series,
            or several pooled).
        tolerance_factor: Tolerance as a multiplicative factor (> 1).
        metric: ``"ale"`` (point error) or ``"crps"`` (probabilistic).
        agg: Aggregation across cutoffs at each horizon.
        window: Smoothing window in months.
    """
    if tolerance_factor <= 1.0:
        raise ValueError("tolerance_factor must be > 1")
    curve = error_curve(scored, metric, agg=agg).set_index("horizon")["error"]
    return first_crossing(curve, float(np.log(tolerance_factor)), metric=metric, window=window)


def band_horizon(
    scored: pd.DataFrame,
    k_efolds: float = 2.0,
    *,
    window: int = 3,
) -> HorizonResult:
    """``h*`` variant: first horizon where the median band width exceeds ``k`` e-folds."""
    curve = error_curve(scored, "band_efolds").set_index("horizon")["error"]
    return first_crossing(curve, float(k_efolds), metric="band_efolds", window=window)
