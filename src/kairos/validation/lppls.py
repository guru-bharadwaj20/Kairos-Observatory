"""Self-contained LPPLS baseline detector for the bubble cases (K-9.03, K-9.04).

The log-periodic power law singularity model (Johansen, Ledoit and Sornette)::

    ln p(t) = A + B·τ^m + C1·τ^m·cos(ω ln τ) + C2·τ^m·sin(ω ln τ),   τ = t_c - t

For fixed ``(t_c, m, ω)`` the four remaining parameters are linear, so the fit
profiles them out with a batched 4×4 normal-equation solve over a grid of the
non-linear parameters (CPU float64), then polishes the best cell with
Nelder-Mead. A fit is *qualified* by the usual filters: ``m ∈ [0.1, 0.9]``,
``ω ∈ [6, 13]``, ``B < 0`` (a bubble, not an anti-bubble) and the damping
condition ``m·|B| ≥ 0.5·ω·|C|``. The search itself is bounded to the ``m`` and
``ω`` ranges.

:class:`LPPLSDetector` repeats the fit over a set of shrinking windows that all
end at the as-of date (the "LPPLS confidence indicator"): confidence is the
share of windows with a qualified fit whose ``t_c`` lies within the horizon,
``t_c`` is their median and the interval their 10-90 % range.

This is the stand-in for the GPU grid search in :mod:`kairos.accel` (K-5.08 to
K-5.11); it implements the same :class:`~kairos.validation.detectors.Detector`
protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import optimize

from kairos.types import SeriesFrame
from kairos.validation.detectors import (
    DETECTORS,
    Prediction,
    decimal_year,
    single_series_id,
)

__all__ = ["LPPLSDetector", "LPPLSFit", "fit_lppls", "lppls_curve"]

FloatArray = npt.NDArray[np.float64]

M_BOUNDS = (0.1, 0.9)
OMEGA_BOUNDS = (6.0, 13.0)
DAMPING_MIN = 0.5
"""Minimum damping ``D = m·|B| / (ω·|C|)`` of a qualified fit."""


@dataclass(frozen=True, slots=True)
class LPPLSFit:
    """One LPPLS fit (all times in decimal years)."""

    t_c: float
    m: float
    omega: float
    a: float
    b: float
    c1: float
    c2: float
    sse: float
    n: int

    @property
    def qualified(self) -> bool:
        """Whether the fit passes the standard LPPLS filters."""
        c = float(np.hypot(self.c1, self.c2))
        return bool(
            M_BOUNDS[0] <= self.m <= M_BOUNDS[1]
            and OMEGA_BOUNDS[0] <= self.omega <= OMEGA_BOUNDS[1]
            and self.b < 0
            and self.m * abs(self.b) >= DAMPING_MIN * self.omega * c
        )


def lppls_curve(t: FloatArray, fit: LPPLSFit) -> FloatArray:
    """Evaluate ``ln p(t)`` for a fit (``t < t_c``)."""
    tau = fit.t_c - t
    tm = tau**fit.m
    phase = fit.omega * np.log(tau)
    return np.asarray(
        fit.a + fit.b * tm + fit.c1 * tm * np.cos(phase) + fit.c2 * tm * np.sin(phase),
        dtype=np.float64,
    )


def _linear_solve(
    t: FloatArray, ly: FloatArray, t_c: FloatArray, m: FloatArray, omega: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Batched least squares for ``(A, B, C1, C2)``; returns ``(coef[G, 4], sse[G])``."""
    tau = t_c[:, None] - t[None, :]
    tm = tau ** m[:, None]
    phase = omega[:, None] * np.log(tau)
    design = np.stack([np.ones_like(tm), tm, tm * np.cos(phase), tm * np.sin(phase)], axis=-1)
    xtx = np.einsum("gni,gnj->gij", design, design)
    xty = np.einsum("gni,n->gi", design, ly)
    scale = np.trace(xtx, axis1=1, axis2=2)[:, None, None] / 4.0
    ridge = 1e-10 * scale * np.eye(4)[None]
    coef = np.linalg.solve(xtx + ridge, xty[..., None])[..., 0]
    resid = ly[None, :] - np.einsum("gni,gi->gn", design, coef)
    return coef, (resid**2).sum(axis=1)


def fit_lppls(
    t: FloatArray,
    y: FloatArray,
    *,
    t_c_max_ahead: float = 1.0,
    n_t_c: int = 24,
    n_m: int = 9,
    n_omega: int = 8,
    polish: bool = True,
) -> LPPLSFit:
    """Fit LPPLS to decimal years ``t`` and raw prices ``y``.

    Args:
        t: Observation times (decimal years), increasing.
        y: Prices, strictly positive.
        t_c_max_ahead: ``t_c`` is searched in ``(t_last, t_last + t_c_max_ahead]``.
        n_t_c: Grid cells for ``t_c``.
        n_m: Grid cells for ``m``.
        n_omega: Grid cells for ``ω``.
        polish: Refine the best grid cell with bounded Nelder-Mead.
    """
    ly = np.log(y)
    t_last = float(t[-1])
    step = max(float(np.median(np.diff(t))), 1e-3) if len(t) > 1 else 1e-3
    tc_grid = t_last + np.linspace(step * 0.5, t_c_max_ahead, n_t_c)
    m_grid = np.linspace(*M_BOUNDS, n_m)
    w_grid = np.linspace(*OMEGA_BOUNDS, n_omega)
    tc_g, m_g, w_g = (a.ravel() for a in np.meshgrid(tc_grid, m_grid, w_grid, indexing="ij"))
    _, sse = _linear_solve(t, ly, tc_g, m_g, w_g)
    best = int(np.nanargmin(sse))
    x0 = np.array([tc_g[best], m_g[best], w_g[best]])

    lo = np.array([t_last + step * 0.25, M_BOUNDS[0], OMEGA_BOUNDS[0]])
    hi = np.array([t_last + t_c_max_ahead, M_BOUNDS[1], OMEGA_BOUNDS[1]])

    def objective(x: FloatArray) -> float:
        x = np.clip(x, lo, hi)
        _, s = _linear_solve(t, ly, x[:1], x[1:2], x[2:3])
        return float(s[0])

    if polish:
        res = optimize.minimize(
            objective,
            x0,
            method="Nelder-Mead",
            bounds=list(zip(lo, hi, strict=True)),
            options={"xatol": 1e-4, "fatol": 1e-10},
        )
        x = np.clip(res.x, lo, hi) if res.fun <= sse[best] else x0
    else:
        x = x0
    coef, s = _linear_solve(t, ly, x[:1], x[1:2], x[2:3])
    a, b, c1, c2 = (float(v) for v in coef[0])
    return LPPLSFit(float(x[0]), float(x[1]), float(x[2]), a, b, c1, c2, float(s[0]), len(t))


@dataclass(frozen=True)
class LPPLSDetector:
    """Multi-window LPPLS confidence detector.

    Attributes:
        name: Detector name.
        windows_years: Window lengths (years back from the as-of date).
        horizon_years: A qualified fit counts only if ``t_c`` is at most this far
            past the last observation.
        alarm_confidence: Share of qualified windows needed for an alarm.
        min_points: Windows with fewer points are skipped.
    """

    name: str = "lppls"
    windows_years: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    horizon_years: float = 0.75
    alarm_confidence: float = 0.3
    min_points: int = 12

    def __call__(self, frame: SeriesFrame, as_of: date) -> Prediction:
        """Fit every window ending at the last observation of ``frame``."""
        series_id = single_series_id(frame)
        data = frame.dropna(subset=["y"]).sort_values("ds")
        t_all = decimal_year(data["ds"])
        y_all = data["y"].to_numpy(dtype=np.float64)
        fits: list[LPPLSFit] = []
        tried = 0
        for window in self.windows_years:
            mask = t_all >= t_all[-1] - window if len(t_all) else np.zeros(0, bool)
            if mask.sum() < self.min_points:
                continue
            tried += 1
            fits.append(fit_lppls(t_all[mask], y_all[mask], t_c_max_ahead=self.horizon_years))
        good = [f for f in fits if f.qualified and f.t_c - t_all[-1] <= self.horizon_years]
        confidence = len(good) / tried if tried else 0.0
        detail: dict[str, Any] = {"confidence": confidence, "windows": tried}
        if not good:
            return Prediction(
                self.name, series_id, as_of, len(t_all), None, None, None, None, False, 0, detail
            )
        tcs = np.array([f.t_c for f in good])
        detail["m"] = float(np.median([f.m for f in good]))
        detail["omega"] = float(np.median([f.omega for f in good]))
        alarm = confidence >= self.alarm_confidence
        level = 0 if not alarm else (2 if confidence >= 0.6 else 1)
        return Prediction(
            detector=self.name,
            series_id=series_id,
            as_of=as_of,
            n_obs=len(t_all),
            t_c=float(np.median(tcs)),
            t_c_lo=float(np.quantile(tcs, 0.1)),
            t_c_hi=float(np.quantile(tcs, 0.9)),
            delta=None,
            alarm=alarm,
            level=level,
            detail=detail,
        )


DETECTORS.setdefault("lppls", LPPLSDetector)
