"""Shrinkage test on ``h*(t)``: Mann-Kendall plus Theil-Sen slope (K-6.06).

Consecutive ``h*(t)`` readings share most of their trailing window, so they
are strongly autocorrelated. The plain Mann-Kendall variance would then
overstate significance. The default is therefore the Hamed & Rao (1998)
variance correction, which uses the autocorrelation of the ranks of the
Sen-detrended series.

H1 is one-sided, "``h*`` is decreasing". The Theil-Sen slope is reported in
months of horizon per year, with its ``1 - alpha`` confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm, rankdata, theilslopes

__all__ = ["ShrinkageResult", "mann_kendall_s", "shrinkage_by_series", "shrinkage_test"]

Correction = Literal["hamed_rao", "none"]


@dataclass(frozen=True, slots=True)
class ShrinkageResult:
    """Result of :func:`shrinkage_test`.

    Attributes:
        n: Readings used.
        s: Mann-Kendall ``S`` statistic.
        tau: Kendall's tau-a, ``S / (n (n - 1) / 2)``.
        z: Continuity-corrected normal score.
        p_value: One-sided p-value for a *decreasing* trend.
        slope: Theil-Sen slope, months of ``h*`` per year.
        slope_lo: Lower ``1 - alpha`` bound of the slope.
        slope_hi: Upper ``1 - alpha`` bound of the slope.
        alpha: Significance level.
        correction: Variance correction used.
        shrinking: ``p_value < alpha`` and ``slope < 0``.
    """

    n: int
    s: float
    tau: float
    z: float
    p_value: float
    slope: float
    slope_lo: float
    slope_hi: float
    alpha: float
    correction: str
    shrinking: bool


def mann_kendall_s(x: npt.ArrayLike) -> tuple[float, float]:
    """Return ``(S, Var(S))`` with the tie correction."""
    arr = np.asarray(x, dtype="float64")
    n = arr.size
    diff = arr[None, :] - arr[:, None]
    s = float(np.sign(diff[np.triu_indices(n, k=1)]).sum())
    _, counts = np.unique(arr, return_counts=True)
    ties = float((counts * (counts - 1) * (2 * counts + 5)).sum())
    var = (n * (n - 1) * (2 * n + 5) - ties) / 18.0
    return s, var


def _hamed_rao_factor(x: npt.NDArray[np.float64], t: npt.NDArray[np.float64]) -> float:
    n = x.size
    slope = theilslopes(x, t).slope
    ranks = rankdata(x - slope * t)
    r = ranks - ranks.mean()
    denom = float((r * r).sum())
    if denom == 0.0:
        return 1.0
    bound = norm.ppf(0.975) / np.sqrt(n)
    total = 0.0
    for k in range(1, n - 2):
        rho = float((r[:-k] * r[k:]).sum() / denom)
        if abs(rho) > bound:
            total += (n - k) * (n - k - 1) * (n - k - 2) * rho
    factor = 1.0 + 2.0 / (n * (n - 1) * (n - 2)) * total
    return max(factor, 1e-6)


def shrinkage_test(
    h_star: pd.Series,
    *,
    alpha: float = 0.05,
    correction: Correction = "hamed_rao",
) -> ShrinkageResult:
    """Test whether ``h*(t)`` is shrinking.

    Args:
        h_star: ``h*`` values indexed by ``as_of`` (datetime-like).
        alpha: One-sided significance level.
        correction: ``"hamed_rao"`` (default; autocorrelation-robust) or ``"none"``.

    Raises:
        ValueError: With fewer than 4 non-missing readings.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    clean = h_star.dropna().sort_index()
    n = len(clean)
    if n < 4:
        raise ValueError("need at least 4 readings for a trend test")
    x = clean.to_numpy("float64")
    idx = pd.DatetimeIndex(clean.index)
    t = ((idx.year - idx[0].year) * 12 + (idx.month - idx[0].month)).to_numpy("float64") / 12.0
    s, var = mann_kendall_s(x)
    if correction == "hamed_rao":
        var *= _hamed_rao_factor(x, t)
    z = 0.0 if var <= 0 else (s - np.sign(s)) / np.sqrt(var)
    p = float(norm.cdf(z))
    sen = theilslopes(x, t, alpha=1 - alpha)
    return ShrinkageResult(
        n=n,
        s=s,
        tau=s / (n * (n - 1) / 2),
        z=float(z),
        p_value=p,
        slope=float(sen.slope),
        slope_lo=float(sen.low_slope),
        slope_hi=float(sen.high_slope),
        alpha=alpha,
        correction=correction,
        shrinking=bool(p < alpha and sen.slope < 0),
    )


def shrinkage_by_series(
    hstar: pd.DataFrame,
    *,
    alpha: float = 0.05,
    correction: Correction = "hamed_rao",
    column: str = "h_star",
) -> pd.DataFrame:
    """Run :func:`shrinkage_test` for every series of an ``h*(t)`` table.

    Series with fewer than 4 readings are skipped.
    """
    rows = []
    for series_id, group in hstar.groupby("series_id", sort=True):
        values = group.set_index("as_of")[column]
        if values.notna().sum() < 4:
            continue
        res = shrinkage_test(values, alpha=alpha, correction=correction)
        rows.append({"series_id": series_id, **{f: getattr(res, f) for f in res.__slots__}})
    return pd.DataFrame(rows)
