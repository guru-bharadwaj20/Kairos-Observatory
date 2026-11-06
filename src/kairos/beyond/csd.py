"""Critical slowing down (Scheffer et al. 2009) as an optional fifth dial.

Near a tipping point a system recovers from small shocks more slowly. In a time
series this shows up as rising lag-1 autocorrelation and rising variance of the
detrended residuals. The dial computes both in a rolling window and tests for an
upward trend with Kendall's tau.

The thresholds are parameters of :class:`CSDConfig`. Deployments read them from
versioned YAML, like every other KAIROS threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["CSDConfig", "CSDDial", "csd_dial", "detrend", "kendall_tau", "rolling_indicators"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CSDConfig:
    """Settings for the critical-slowing-down dial.

    Attributes:
        window: Rolling window as a fraction of the series length.
        bandwidth: Gaussian smoothing bandwidth for detrending, as a fraction of
            the series length.
        tau_threshold: Kendall's tau above which an indicator counts as rising.
    """

    window: float = 0.5
    bandwidth: float = 0.1
    tau_threshold: float = 0.5


@dataclass(frozen=True)
class CSDDial:
    """Result of the dial: indicator series, their trends and the state."""

    ar1: FloatArray
    variance: FloatArray
    tau_ar1: float
    tau_variance: float
    state: Literal["calm", "danger"]


def detrend(x: ArrayLike, bandwidth: float) -> FloatArray:
    """Subtract a Gaussian kernel smooth with ``bandwidth`` given in samples."""
    y = np.asarray(x, dtype=np.float64)
    idx = np.arange(y.size, dtype=np.float64)
    w = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / bandwidth) ** 2)
    smooth = (w @ y) / w.sum(axis=1)
    return np.asarray(y - smooth, dtype=np.float64)


def rolling_indicators(resid: ArrayLike, window: int) -> tuple[FloatArray, FloatArray]:
    """Rolling lag-1 autocorrelation and variance; the first ``window - 1`` are NaN."""
    r = np.asarray(resid, dtype=np.float64)
    if not 3 <= window <= r.size:
        raise ValueError(f"window must be in [3, {r.size}], got {window}")
    ar1 = np.full(r.size, np.nan)
    var = np.full(r.size, np.nan)
    for end in range(window, r.size + 1):
        seg = r[end - window : end]
        seg = seg - seg.mean()
        denom = float(np.dot(seg, seg))
        ar1[end - 1] = float(np.dot(seg[:-1], seg[1:]) / denom) if denom > 0 else np.nan
        var[end - 1] = denom / (window - 1)
    return ar1, var


def kendall_tau(y: ArrayLike) -> float:
    """Kendall's tau-a of ``y`` against time, ignoring NaNs."""
    v = np.asarray(y, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return float("nan")
    s = np.sign(v[None, :] - v[:, None])
    concordant = float(np.triu(s, k=1).sum())
    return float(concordant / (n * (n - 1) / 2))


def csd_dial(x: ArrayLike, config: CSDConfig | None = None) -> CSDDial:
    """Compute the critical-slowing-down dial for one series.

    The state is ``danger`` when both lag-1 autocorrelation and variance have a
    Kendall's tau above ``config.tau_threshold``.
    """
    cfg = config or CSDConfig()
    y = np.asarray(x, dtype=np.float64)
    resid = detrend(y, max(1.0, cfg.bandwidth * y.size))
    ar1, var = rolling_indicators(resid, max(3, round(cfg.window * y.size)))
    t_ar1, t_var = kendall_tau(ar1), kendall_tau(var)
    danger = t_ar1 > cfg.tau_threshold and t_var > cfg.tau_threshold
    return CSDDial(ar1, var, t_ar1, t_var, "danger" if danger else "calm")
