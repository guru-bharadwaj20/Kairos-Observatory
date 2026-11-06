"""Growth-rate estimator and the ``δ`` regression (K-5.02, design notes D1 and D5).

``g(t) = d ln x/dt`` is estimated from the **historical data only** with a cubic
smoothing spline (:func:`scipy.interpolate.make_smoothing_spline`) whose penalty ``λ``
is chosen by generalised cross-validation (GCV). Prophet's piecewise-linear trend is
never differentiated (D1): its ``g`` is a step function and biases ``δ`` towards 0.

For fixed ``λ`` the spline is a linear smoother, so its fitted values and derivative
are matrices applied to ``ln y``: ``ln x̂ = S·ln y`` and ``ĝ = D·ln y``
(:class:`SplineSmoother`). The bootstrap (K-5.04/K-5.05) reuses these operators.

The acceleration exponent comes from the regression

    ln g = ln k + δ · ln x

(``δ = 0`` exponential, ``δ > 0`` hyperbolic, ``δ < 0`` saturating). For ``δ > 0`` the
time left to the singularity is ``T_remaining = t_c − t = 1/(δ·g)`` (D5), evaluated at
the last observation with the regression-implied ``g``; its delta-method interval uses
the OLS covariance of ``(ln k, δ)``. OLS standard errors ignore serial correlation and
errors in variables (D4), so they are too narrow; use :mod:`kairos.accel.bootstrap`
for reported intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.interpolate import make_smoothing_spline

from kairos.accel.synthetic import FloatArray, decimal_years
from kairos.types import SeriesFrame

__all__ = [
    "DeltaFit",
    "SplineSmoother",
    "delta_from_frame",
    "edge_mask",
    "estimate_delta",
    "frame_arrays",
    "ols_delta",
    "select_lambda",
    "spline_smoother",
]

_LOG10_LAM_BOUNDS = (-8.0, 6.0)


@dataclass(frozen=True)
class SplineSmoother:
    """A cubic smoothing spline with fixed penalty, as linear operators.

    Attributes:
        t: Knots / evaluation times (decimal years, strictly increasing).
        lam: Roughness penalty ``λ``.
        S: Hat matrix, ``S @ z`` is the smoothed series.
        D: Derivative matrix, ``D @ z`` is the spline's first derivative at ``t``.
    """

    t: FloatArray
    lam: float
    S: FloatArray
    D: FloatArray

    @property
    def edf(self) -> float:
        """Effective degrees of freedom, ``trace(S)``."""
        return float(np.trace(self.S))

    def smooth(self, z: FloatArray) -> FloatArray:
        """Smoothed values (``z`` may be ``(n,)`` or ``(B, n)``)."""
        out: FloatArray = z @ self.S.T
        return out

    def derivative(self, z: FloatArray) -> FloatArray:
        """First derivative of the smoothing spline (``z`` may be ``(n,)`` or ``(B, n)``)."""
        out: FloatArray = z @ self.D.T
        return out


def _check_t(t: FloatArray) -> FloatArray:
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or len(t) < 8:
        raise ValueError("need a 1-D time vector with at least 8 points")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")
    return t


def spline_smoother(t: FloatArray, lam: float) -> SplineSmoother:
    """Build the smoother and derivative operators of a cubic smoothing spline.

    Args:
        t: Strictly increasing times.
        lam: Penalty ``λ > 0``.
    """
    t = _check_t(t)
    if not lam > 0:
        raise ValueError("lam must be > 0")
    spl = make_smoothing_spline(t, np.eye(len(t)), lam=lam)
    # Fitting the identity: column j is the response to the j-th unit vector.
    s_mat = np.asarray(spl(t), dtype=np.float64)
    d_mat = np.asarray(spl.derivative()(t), dtype=np.float64)
    return SplineSmoother(t=t, lam=float(lam), S=s_mat, D=d_mat)


def _gcv(t: FloatArray, z: FloatArray, log10_lam: float) -> float:
    sm = spline_smoother(t, 10.0**log10_lam)
    resid = z - sm.smooth(z)
    n = len(z)
    denom = (1.0 - sm.edf / n) ** 2
    return float(np.mean(resid**2) / max(denom, 1e-12))


def select_lambda(t: FloatArray, z: FloatArray) -> float:
    """Choose the spline penalty by generalised cross-validation.

    A coarse grid over ``log10 λ ∈ [-8, 6]`` is followed by a bounded scalar search
    around the best grid point.

    Args:
        t: Strictly increasing times.
        z: Values to smooth (``ln y``).
    """
    t = _check_t(t)
    z = np.asarray(z, dtype=np.float64)
    grid = np.linspace(*_LOG10_LAM_BOUNDS, 29)
    scores = np.array([_gcv(t, z, g) for g in grid])
    i = int(np.argmin(scores))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = optimize.minimize_scalar(
        lambda v: _gcv(t, z, float(v)), bounds=(lo, hi), method="bounded"
    )
    best = float(res.x) if res.fun <= scores[i] else float(grid[i])
    return float(10.0**best)


def edge_mask(n: int, edge_trim: int) -> FloatArray:
    """Weights that drop ``edge_trim`` points at each end (spline derivatives are biased there)."""
    if 2 * edge_trim >= n:
        raise ValueError("edge_trim leaves no points")
    w = np.ones(n, dtype=np.float64)
    if edge_trim:
        w[:edge_trim] = 0.0
        w[-edge_trim:] = 0.0
    return w


def ols_delta(
    log_x: FloatArray, g: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Row-wise weighted OLS of ``ln g`` on ``ln x`` over points with ``g > 0``.

    Args:
        log_x: ``(B, n)`` or ``(n,)`` smoothed log levels.
        g: Growth rates, same shape.
        weights: ``(n,)`` base weights (edge mask); points with ``g <= 0`` get 0.

    Returns:
        ``(intercept, delta, n_used)`` arrays of shape ``(B,)`` (or scalars as 0-d
        arrays). Rows with fewer than 3 usable points give ``nan``.
    """
    w = np.where(g > 0, weights, 0.0)
    lg = np.log(np.where(g > 0, g, 1.0))
    sw = w.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = (w * log_x).sum(axis=-1) / sw
        my = (w * lg).sum(axis=-1) / sw
        dx = log_x - mx[..., None]
        sxx = (w * dx * dx).sum(axis=-1)
        sxy = (w * dx * (lg - my[..., None])).sum(axis=-1)
        delta = sxy / sxx
        intercept = my - delta * mx
    bad = sw < 3
    delta = np.where(bad, np.nan, delta)
    intercept = np.where(bad, np.nan, intercept)
    return intercept, delta, sw


def t_remaining(intercept: FloatArray, delta: FloatArray, log_x_last: FloatArray) -> FloatArray:
    """``T = 1/(δ·g_last)`` with ``g_last = exp(ln k + δ ln x_last)``; ``nan`` where ``δ <= 0``."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        g_last = np.exp(intercept + delta * log_x_last)
        out = 1.0 / (delta * g_last)
    result: FloatArray = np.where(delta > 0, out, np.nan)
    return result


@dataclass(frozen=True)
class DeltaFit:
    """Result of :func:`estimate_delta`.

    Attributes:
        delta: Acceleration exponent ``δ``.
        delta_se: OLS standard error of ``δ`` (too small under autocorrelation, D4).
        intercept: ``ln k``.
        cov: 2×2 OLS covariance of ``(ln k, δ)``.
        r2: R² of the regression.
        n_used: Points used (edge-trimmed, ``g > 0``).
        lam: Spline penalty used.
        edf: Effective degrees of freedom of the spline.
        edge_trim: Points dropped at each end.
        t: Times.
        log_x: Smoothed ``ln x``.
        g: Growth rate ``d ln x/dt`` per year.
        t_last: Last observation time.
        t_remaining: ``1/(δ·g)`` at ``t_last`` in years (``nan`` unless ``δ > 0``).
        t_remaining_se: Delta-method standard error of ``t_remaining``.
        t_c: ``t_last + t_remaining``.
        t_c_ci: Delta-method interval for ``t_c`` at ``level``.
        level: Interval coverage.
    """

    delta: float
    delta_se: float
    intercept: float
    cov: FloatArray
    r2: float
    n_used: int
    lam: float
    edf: float
    edge_trim: int
    t: FloatArray
    log_x: FloatArray
    g: FloatArray
    t_last: float
    t_remaining: float
    t_remaining_se: float
    t_c: float
    t_c_ci: tuple[float, float]
    level: float

    @property
    def delta_ci(self) -> tuple[float, float]:
        """Normal-theory OLS interval for ``δ`` at ``level`` (optimistic, see D4)."""
        z = float(stats.norm.ppf(0.5 + self.level / 2))
        return (self.delta - z * self.delta_se, self.delta + z * self.delta_se)


def default_edge_trim(n: int) -> int:
    """Default edge trim: 5 % of the points at each end, at least one."""
    return max(1, round(0.05 * n))


def estimate_delta(
    t: FloatArray,
    log_y: FloatArray,
    *,
    lam: float | None = None,
    edge_trim: int | None = None,
    level: float = 0.9,
) -> DeltaFit:
    """Estimate ``δ``, ``T_remaining`` and ``t_c`` from one historical series.

    Args:
        t: Strictly increasing decimal-year times.
        log_y: Observed ``ln y`` (history only; never forecasts, D1).
        lam: Spline penalty; ``None`` selects it by GCV.
        edge_trim: Points dropped at each end of the regression; default 5 %.
        level: Coverage of the delta-method interval.

    Raises:
        ValueError: If fewer than 3 points have ``g > 0``.
    """
    t = _check_t(t)
    z = np.asarray(log_y, dtype=np.float64)
    if z.shape != t.shape or not np.all(np.isfinite(z)):
        raise ValueError("log_y must be finite and match t")
    if lam is None:
        lam = select_lambda(t, z)
    sm = spline_smoother(t, lam)
    trim = default_edge_trim(len(t)) if edge_trim is None else edge_trim
    log_x, g = sm.smooth(z), sm.derivative(z)
    w = edge_mask(len(t), trim)
    a_arr, d_arr, n_arr = ols_delta(log_x, g, w)
    a, d, n_used = float(a_arr), float(d_arr), int(n_arr)
    if not math.isfinite(d):
        raise ValueError("fewer than 3 points with positive growth; δ is undefined")

    use = (w > 0) & (g > 0)
    x, y = log_x[use], np.log(g[use])
    resid = y - (a + d * x)
    sigma2 = float(resid @ resid) / max(n_used - 2, 1)
    xmat = np.column_stack([np.ones_like(x), x])
    cov = sigma2 * np.linalg.inv(xmat.T @ xmat)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")

    lx_last = float(log_x[-1])
    t_rem = float(t_remaining(np.asarray(a), np.asarray(d), np.asarray(lx_last)))
    if math.isfinite(t_rem):
        grad = np.array([-t_rem, -t_rem * (1.0 / d + lx_last)])
        t_rem_se = float(math.sqrt(max(grad @ cov @ grad, 0.0)))
        zq = float(stats.norm.ppf(0.5 + level / 2))
        t_c = float(t[-1]) + t_rem
        t_c_ci = (t_c - zq * t_rem_se, t_c + zq * t_rem_se)
    else:
        t_rem_se, t_c, t_c_ci = float("nan"), float("nan"), (float("nan"), float("nan"))

    return DeltaFit(
        delta=d,
        delta_se=float(math.sqrt(cov[1, 1])),
        intercept=a,
        cov=cov,
        r2=r2,
        n_used=n_used,
        lam=float(lam),
        edf=sm.edf,
        edge_trim=trim,
        t=t,
        log_x=log_x,
        g=g,
        t_last=float(t[-1]),
        t_remaining=t_rem,
        t_remaining_se=t_rem_se,
        t_c=t_c,
        t_c_ci=t_c_ci,
        level=level,
    )


def frame_arrays(frame: SeriesFrame, series_id: str | None = None) -> tuple[FloatArray, FloatArray]:
    """Decimal-year times and ``ln y`` of one series in a ``SeriesFrame``, sorted by ``ds``.

    Args:
        frame: Long-format frame; rows with missing ``y`` are dropped.
        series_id: Which series; may be omitted when the frame holds just one.
    """
    if series_id is None:
        ids = frame["series_id"].unique()
        if len(ids) != 1:
            raise ValueError(f"frame holds {len(ids)} series; pass series_id")
        series_id = str(ids[0])
    sub = frame.loc[frame["series_id"] == series_id, ["ds", "y"]].dropna().sort_values("ds")
    if (sub["y"] <= 0).any():
        raise ValueError("y must be positive")
    t = decimal_years(sub["ds"])
    log_y: FloatArray = np.log(sub["y"].to_numpy(dtype=np.float64))
    return t, log_y


def delta_from_frame(
    frame: SeriesFrame,
    series_id: str | None = None,
    *,
    lam: float | None = None,
    edge_trim: int | None = None,
    level: float = 0.9,
) -> DeltaFit:
    """:func:`estimate_delta` on one series of a ``SeriesFrame``."""
    t, log_y = frame_arrays(frame, series_id)
    return estimate_delta(t, log_y, lam=lam, edge_trim=edge_trim, level=level)
