"""Direct nonlinear fit of the finite-time singularity law (K-5.03, design note D1).

Cross-check for the spline/regression estimator in :mod:`kairos.accel.growth`. Fits

    x(t) = x0 · (1 − t/t_c)^(−1/δ)

in log space (multiplicative noise), written with a free intercept as

    ln x = a − (1/δ) · ln(t_c − t),

by bounded :func:`scipy.optimize.least_squares` over ``(a, δ, t_c)`` with
``t_c > t_last``. Starting values come from :func:`kairos.accel.growth.estimate_delta`
when it finds ``δ > 0``; otherwise a small grid over ``t_c`` seeds the search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from kairos.accel.growth import estimate_delta, frame_arrays
from kairos.accel.synthetic import FloatArray
from kairos.types import SeriesFrame

__all__ = ["HyperbolicFit", "fit_hyperbolic", "fit_hyperbolic_frame", "hyperbolic_log_curve"]

_DELTA_BOUNDS = (1e-3, 50.0)


def hyperbolic_log_curve(t: FloatArray, a: float, delta: float, t_c: float) -> FloatArray:
    """``a − (1/δ)·ln(t_c − t)``; requires ``t < t_c``."""
    out: FloatArray = a - np.log(t_c - np.asarray(t, dtype=np.float64)) / delta
    return out


@dataclass(frozen=True)
class HyperbolicFit:
    """Result of :func:`fit_hyperbolic`.

    Attributes:
        a: Intercept of ``ln x``.
        delta: Exponent ``δ``.
        t_c: Critical time (decimal years).
        se: Asymptotic standard errors of ``(a, δ, t_c)`` from the Jacobian.
        rmse: Root-mean-square log residual.
        success: Optimiser convergence flag.
        at_bound: Whether ``δ`` or ``t_c`` ended on a bound (the fit is then unreliable).
        level: Coverage of :attr:`t_c_ci` and :attr:`delta_ci`.
    """

    a: float
    delta: float
    t_c: float
    se: tuple[float, float, float]
    rmse: float
    success: bool
    at_bound: bool
    level: float = 0.9

    def _ci(self, value: float, se: float) -> tuple[float, float]:
        z = float(stats.norm.ppf(0.5 + self.level / 2))
        return (value - z * se, value + z * se)

    @property
    def delta_ci(self) -> tuple[float, float]:
        """Wald interval for ``δ`` (assumes independent errors)."""
        return self._ci(self.delta, self.se[1])

    @property
    def t_c_ci(self) -> tuple[float, float]:
        """Wald interval for ``t_c`` (assumes independent errors)."""
        return self._ci(self.t_c, self.se[2])

    def predict(self, t: FloatArray) -> FloatArray:
        """Fitted ``ln x`` at ``t`` (``t < t_c``)."""
        return hyperbolic_log_curve(t, self.a, self.delta, self.t_c)


def _profile_start(t: FloatArray, z: FloatArray, t_c: float) -> tuple[float, float, float]:
    """Best ``(a, δ)`` by linear OLS of ``z`` on ``ln(t_c − t)`` for fixed ``t_c``; returns SSE."""
    x = np.log(t_c - t)
    slope, intercept = np.polyfit(x, z, 1)
    delta = -1.0 / slope if slope < 0 else _DELTA_BOUNDS[1]
    resid = z - (intercept + slope * x)
    return float(intercept), float(np.clip(delta, *_DELTA_BOUNDS)), float(resid @ resid)


def fit_hyperbolic(
    t: FloatArray,
    log_y: FloatArray,
    *,
    t_c_max_ahead: float | None = None,
    level: float = 0.9,
) -> HyperbolicFit:
    """Fit ``ln y = a − (1/δ) ln(t_c − t)`` by bounded nonlinear least squares.

    Args:
        t: Strictly increasing decimal-year times.
        log_y: Observed ``ln y``.
        t_c_max_ahead: Upper bound on ``t_c − t_last`` in years; default ten times the
            observed span.
        level: Coverage of the Wald intervals.
    """
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(log_y, dtype=np.float64)
    if t.shape != z.shape or len(t) < 5:
        raise ValueError("need matching t and log_y with at least 5 points")
    span = float(t[-1] - t[0])
    ahead = 10.0 * span if t_c_max_ahead is None else t_c_max_ahead
    eps = 1e-6 * max(span, 1.0)
    lo = np.array([-np.inf, _DELTA_BOUNDS[0], t[-1] + eps])
    hi = np.array([np.inf, _DELTA_BOUNDS[1], t[-1] + ahead])

    starts: list[float] = []
    try:
        guess = estimate_delta(t, z)
        if guess.delta > 0 and math.isfinite(guess.t_c):
            starts.append(float(np.clip(guess.t_c, lo[2] + eps, hi[2] - eps)))
    except ValueError:
        pass
    starts += list(t[-1] + np.geomspace(0.01 * span, ahead, 12))
    best_tc = min(starts, key=lambda tc: _profile_start(t, z, tc)[2])
    a0, d0, _ = _profile_start(t, z, best_tc)
    x0 = np.clip(np.array([a0, d0, best_tc]), lo + eps, hi - eps)

    def resid(p: FloatArray) -> FloatArray:
        return hyperbolic_log_curve(t, p[0], p[1], p[2]) - z

    res = optimize.least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", method="trf")
    p = res.x
    n = len(t)
    dof = max(n - 3, 1)
    s2 = float(res.fun @ res.fun) / dof
    jac = res.jac
    try:
        cov = s2 * np.linalg.inv(jac.T @ jac)
        se_arr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se_arr = np.full(3, np.nan)
    tol = 1e-6
    at_bound = bool(
        np.any(np.abs(p[1:] - lo[1:]) <= tol * np.maximum(1.0, np.abs(lo[1:])))
        or np.any(np.abs(p[1:] - hi[1:]) <= tol * np.maximum(1.0, np.abs(hi[1:])))
    )
    return HyperbolicFit(
        a=float(p[0]),
        delta=float(p[1]),
        t_c=float(p[2]),
        se=(float(se_arr[0]), float(se_arr[1]), float(se_arr[2])),
        rmse=float(math.sqrt(float(res.fun @ res.fun) / n)),
        success=bool(res.success),
        at_bound=at_bound,
        level=level,
    )


def fit_hyperbolic_frame(
    frame: SeriesFrame, series_id: str | None = None, *, level: float = 0.9
) -> HyperbolicFit:
    """:func:`fit_hyperbolic` on one series of a ``SeriesFrame``."""
    t, log_y = frame_arrays(frame, series_id)
    return fit_hyperbolic(t, log_y, level=level)
