"""LPPLS reference fit on the CPU in float64 (K-5.07).

Log-periodic power law singularity, Filimonov–Sornette (2013) linearised form:

    ln p(t) = A + B·f + C1·f·cos(ω ln τ) + C2·f·sin(ω ln τ),   τ = t_c − t,  f = τ^m

For fixed nonlinear parameters ``(t_c, m, ω)`` the four linear parameters
``(A, B, C1, C2)`` follow from ordinary least squares, so the fit is a 3-D search over
the profile SSE: a grid search followed by bounded Nelder–Mead polishing.

The linear solve uses **centred** columns (subtract the mean of ``ln p`` and of the
three basis columns, solve the 3×3 normal equations, recover ``A``). The SSE does not
change when time is rescaled, so ``τ`` is measured in units of the window length;
reported ``t_c`` is in decimal years. This is the float64 reference for the batched
GPU search in :mod:`kairos.accel.lppls_gpu` (K-5.09); only ``t_c > t2`` is searched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from kairos.accel.synthetic import FloatArray, SyntheticSeries, _finish, decimal_years, month_starts

__all__ = [
    "DEFAULT_M_RANGE",
    "DEFAULT_OMEGA_RANGE",
    "DEFAULT_TC_RANGE",
    "LPPLSFit",
    "fit_lppls",
    "lppls_curve",
    "lppls_linear",
    "lppls_series",
    "lppls_sse_cells",
    "make_grid",
    "polish",
]

DEFAULT_TC_RANGE = (0.002, 0.5)
"""Search range of ``(t_c − t2)/(t2 − t1)``."""
DEFAULT_M_RANGE = (0.01, 0.99)
DEFAULT_OMEGA_RANGE = (2.0, 25.0)


def lppls_curve(
    t: FloatArray,
    tc: float,
    m: float,
    omega: float,
    a: float,
    b: float,
    c1: float,
    c2: float,
) -> FloatArray:
    """``ln p(t)`` of the LPPLS model (requires ``t < tc``)."""
    tau = tc - np.asarray(t, dtype=np.float64)
    f = tau**m
    lt = np.log(tau)
    out: FloatArray = a + b * f + c1 * f * np.cos(omega * lt) + c2 * f * np.sin(omega * lt)
    return out


def lppls_series(
    n: int = 240,
    *,
    start: str = "2000-01-01",
    tc: float = 2021.0,
    m: float = 0.5,
    omega: float = 8.0,
    a: float = 5.0,
    b: float = -1.0,
    c1: float = 0.05,
    c2: float = 0.05,
    noise: float = 0.01,
    ar: float = 0.0,
    seed: int = 0,
    series_id: str | None = None,
    source_group: str = "synthetic",
) -> SyntheticSeries:
    """Synthetic LPPLS bubble with known ``t_c``, ``m`` and ``ω`` (multiplicative noise)."""
    ds = month_starts(start, n)
    t = decimal_years(ds)
    if t[-1] >= tc:
        raise ValueError("tc must lie after the last point")
    log_x = lppls_curve(t, tc, m, omega, a, b, c1, c2)
    return _finish(
        "lppls",
        ds,
        t,
        log_x,
        noise=noise,
        ar=ar,
        seed=seed,
        series_id=series_id,
        source_group=source_group,
        params={"t_c": tc, "m": m, "omega": omega, "a": a, "b": b, "c1": c1, "c2": c2},
    )


def lppls_linear(
    t: FloatArray, log_p: FloatArray, tc: float, m: float, omega: float
) -> tuple[FloatArray, float]:
    """Linear parameters ``(A, B, C1, C2)`` and SSE for fixed ``(tc, m, ω)`` by ``lstsq``."""
    tau = tc - np.asarray(t, dtype=np.float64)
    if np.any(tau <= 0):
        raise ValueError("tc must lie after every t")
    f = tau**m
    lt = np.log(tau)
    x = np.column_stack([np.ones_like(f), f, f * np.cos(omega * lt), f * np.sin(omega * lt)])
    coef, *_ = np.linalg.lstsq(x, log_p, rcond=None)
    resid = log_p - x @ coef
    return coef, float(resid @ resid)


def lppls_sse_cells(
    t: FloatArray,
    log_p: FloatArray,
    tc: FloatArray,
    m: FloatArray,
    omega: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Profile SSE and linear parameters for many ``(tc, m, ω)`` cells, float64.

    Uses the centred normal equations, the same algorithm as the GPU kernel, with
    ``τ`` scaled by the window length.

    Args:
        t: Window times (decimal years).
        log_p: ``ln p`` on the window.
        tc: ``(K,)`` critical times, each ``> t[-1]``.
        m: ``(K,)`` exponents.
        omega: ``(K,)`` angular log-frequencies.

    Returns:
        ``sse`` of shape ``(K,)`` and ``coef`` of shape ``(K, 4)`` (``A, B, C1, C2`` in
        original time units).
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(log_p, dtype=np.float64)
    tc, m, omega = (np.asarray(v, dtype=np.float64) for v in (tc, m, omega))
    span = float(t[-1] - t[0]) or 1.0
    tau = (tc[:, None] - t[None, :]) / span
    if np.any(tau <= 0):
        raise ValueError("tc must lie after every t")
    lt = np.log(tau)
    f = tau ** m[:, None]
    basis = np.stack([f, f * np.cos(omega[:, None] * lt), f * np.sin(omega[:, None] * lt)], -1)
    ym = y.mean()
    yc = y - ym
    bm = basis.mean(axis=1, keepdims=True)
    bc = basis - bm
    xtx = np.einsum("kni,knj->kij", bc, bc)
    xty = np.einsum("kni,n->ki", bc, yc)
    ridge = 1e-12 * np.trace(xtx, axis1=1, axis2=2)[:, None, None] * np.eye(3)
    beta = np.linalg.solve(xtx + ridge, xty[..., None])[..., 0]
    resid = yc[None, :] - np.einsum("kni,ki->kn", bc, beta)
    sse = np.einsum("kn,kn->k", resid, resid)
    a = ym - np.einsum("ki,ki->k", bm[:, 0, :], beta)
    coef = _unscale(np.column_stack([a, beta]), m, omega, span)
    return sse, coef


def _unscale(coef: FloatArray, m: FloatArray, omega: FloatArray, span: float) -> FloatArray:
    """Convert coefficients fitted with ``τ/span`` to raw ``τ`` (decimal years)."""
    # f_s = (τ/s)^m = s^-m f ;  ω ln(τ/s) = ω ln τ − ω ln s  (a rotation of C1, C2)
    scale = span ** (-m)
    phi = omega * math.log(span)
    b = coef[:, 1] * scale
    c1s, c2s = coef[:, 2] * scale, coef[:, 3] * scale
    c1 = c1s * np.cos(phi) - c2s * np.sin(phi)
    c2 = c1s * np.sin(phi) + c2s * np.cos(phi)
    out: FloatArray = np.column_stack([coef[:, 0], b, c1, c2])
    return out


def make_grid(
    t1: float,
    t2: float,
    *,
    n_tc: int = 24,
    n_m: int = 12,
    n_omega: int = 12,
    tc_range: tuple[float, float] = DEFAULT_TC_RANGE,
    m_range: tuple[float, float] = DEFAULT_M_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Flattened ``tc × m × ω`` grid for a window ``[t1, t2]``; returns three ``(K,)`` arrays."""
    dt = t2 - t1
    tcs = t2 + dt * np.linspace(*tc_range, n_tc)
    ms = np.linspace(*m_range, n_m)
    oms = np.linspace(*omega_range, n_omega)
    g = np.meshgrid(tcs, ms, oms, indexing="ij")
    return g[0].ravel(), g[1].ravel(), g[2].ravel()


@dataclass(frozen=True)
class LPPLSFit:
    """A fitted LPPLS model on one window.

    Attributes:
        tc: Critical time (decimal years).
        m: Power-law exponent.
        omega: Log-periodic angular frequency.
        a: ``A``.
        b: ``B`` (negative for a positive bubble).
        c1: ``C1``.
        c2: ``C2``.
        sse: Sum of squared log residuals.
        t1: Window start.
        t2: Window end.
        n: Points in the window.
    """

    tc: float
    m: float
    omega: float
    a: float
    b: float
    c1: float
    c2: float
    sse: float
    t1: float
    t2: float
    n: int

    @property
    def c(self) -> float:
        """Oscillation amplitude ``C = sqrt(C1² + C2²)``."""
        return math.hypot(self.c1, self.c2)

    @property
    def rmse(self) -> float:
        """Root-mean-square log residual."""
        return math.sqrt(self.sse / self.n)

    @property
    def damping(self) -> float:
        """Damping ratio ``m|B| / (ω|C|)``."""
        denom = self.omega * self.c
        return math.inf if denom == 0 else self.m * abs(self.b) / denom

    @property
    def n_oscillations(self) -> float:
        """Oscillations between ``t1`` and ``t2``: ``(ω/2π) ln((tc − t1)/(tc − t2))``."""
        return self.omega / (2 * math.pi) * math.log((self.tc - self.t1) / (self.tc - self.t2))

    def predict(self, t: FloatArray) -> FloatArray:
        """Fitted ``ln p`` at ``t``."""
        return lppls_curve(t, self.tc, self.m, self.omega, self.a, self.b, self.c1, self.c2)


def _make_fit(t: FloatArray, y: FloatArray, tc: float, m: float, omega: float) -> LPPLSFit:
    coef, sse = lppls_linear(t, y, tc, m, omega)
    return LPPLSFit(
        tc=float(tc),
        m=float(m),
        omega=float(omega),
        a=float(coef[0]),
        b=float(coef[1]),
        c1=float(coef[2]),
        c2=float(coef[3]),
        sse=sse,
        t1=float(t[0]),
        t2=float(t[-1]),
        n=len(t),
    )


def polish(
    t: FloatArray,
    log_p: FloatArray,
    start: tuple[float, float, float],
    *,
    tc_range: tuple[float, float] = DEFAULT_TC_RANGE,
    m_range: tuple[float, float] = DEFAULT_M_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
) -> LPPLSFit:
    """Bounded Nelder–Mead on the profile SSE from ``start = (tc, m, ω)``, float64."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(log_p, dtype=np.float64)
    dt = float(t[-1] - t[0])
    bounds = [
        (t[-1] + dt * tc_range[0], t[-1] + dt * tc_range[1]),
        m_range,
        omega_range,
    ]
    scale = np.array([dt, 1.0, 10.0])

    def sse(z: FloatArray) -> float:
        tc, m, om = z * scale
        if tc <= t[-1]:
            return math.inf
        return lppls_linear(t, y, tc, m, om)[1]

    x0 = np.clip(
        np.asarray(start, dtype=np.float64), [b[0] for b in bounds], [b[1] for b in bounds]
    )
    sb = [(lo / s, hi / s) for (lo, hi), s in zip(bounds, scale, strict=True)]
    res = optimize.minimize(
        sse,
        x0 / scale,
        method="Nelder-Mead",
        bounds=sb,
        options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 4000},
    )
    best = res.x * scale if res.fun <= sse(x0 / scale) else x0
    return _make_fit(t, y, float(best[0]), float(best[1]), float(best[2]))


def fit_lppls(
    t: FloatArray,
    log_p: FloatArray,
    *,
    n_tc: int = 24,
    n_m: int = 12,
    n_omega: int = 12,
    tc_range: tuple[float, float] = DEFAULT_TC_RANGE,
    m_range: tuple[float, float] = DEFAULT_M_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
) -> LPPLSFit:
    """Fit LPPLS to one window: float64 grid search, then Nelder–Mead polishing.

    Args:
        t: Window times (decimal years, increasing).
        log_p: ``ln p`` on the window.
        n_tc: Grid points for ``t_c``.
        n_m: Grid points for ``m``.
        n_omega: Grid points for ``ω``.
        tc_range: Range of ``(t_c − t2)/(t2 − t1)``.
        m_range: Range of ``m``.
        omega_range: Range of ``ω``.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(log_p, dtype=np.float64)
    if t.shape != y.shape or len(t) < 8:
        raise ValueError("need matching t and log_p with at least 8 points")
    tc, m, om = make_grid(
        float(t[0]),
        float(t[-1]),
        n_tc=n_tc,
        n_m=n_m,
        n_omega=n_omega,
        tc_range=tc_range,
        m_range=m_range,
        omega_range=omega_range,
    )
    sse, _ = lppls_sse_cells(t, y, tc, m, om)
    k = int(np.nanargmin(sse))
    return polish(
        t,
        y,
        (float(tc[k]), float(m[k]), float(om[k])),
        tc_range=tc_range,
        m_range=m_range,
        omega_range=omega_range,
    )
