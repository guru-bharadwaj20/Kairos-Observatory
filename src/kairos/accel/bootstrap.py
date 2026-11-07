"""Moving block bootstrap for ``δ`` and ``t_c`` on the CPU pool (K-5.04, design note D4).

OLS standard errors from the ``ln g`` on ``ln x`` regression are far too small: both
variables come from the same autocorrelated series. The reported uncertainty instead
comes from a residual **moving block bootstrap** (Künsch 1989):

1. Smooth ``ln y`` with the spline of :mod:`kairos.accel.growth` (penalty ``λ`` by GCV)
   to get ``ln x̂`` and residuals ``r = ln y − ln x̂`` (centred).
2. Draw ``⌈n/L⌉`` blocks of ``L`` consecutive residuals with replacement and
   concatenate them to length ``n``; ``ln y* = ln x̂ + r*``.
3. **Refit the smoother** on ``ln y*`` (same ``λ`` by default, or re-select it by GCV
   with ``reselect_lambda=True``), recompute ``g*``, ``δ*`` and
   ``t_c* = t_last + 1/(δ*·g*_last)``.

Draws are split into chunks and run with :func:`kairos.compute.pmap`; each chunk gets
its own seed from :func:`kairos.compute.derive_seed`, so results do not depend on the
worker count. ``t_c*`` is ``nan`` for draws with ``δ* <= 0`` (no singularity); interval
helpers treat those as ``+∞``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from kairos.accel.growth import (
    DeltaFit,
    edge_mask,
    estimate_delta,
    frame_arrays,
    ols_delta,
    spline_smoother,
    t_remaining,
)
from kairos.accel.synthetic import FloatArray
from kairos.compute import derive_seed, pmap
from kairos.types import SeriesFrame

__all__ = [
    "BootstrapResult",
    "block_bootstrap",
    "block_bootstrap_frame",
    "default_block_len",
    "mbb_indices",
]

IntArray = npt.NDArray[np.int64]
Quantity = Literal["delta", "t_c", "t_remaining"]


def default_block_len(n: int) -> int:
    """Block length rule of thumb ``L ≈ n^(1/3)`` (Hall, Horowitz and Jing 1995), at least 2."""
    blen: int = math.floor(float(n) ** (1.0 / 3.0) + 0.5)
    return max(2, blen)


def mbb_indices(n: int, block_len: int, n_draws: int, rng: np.random.Generator) -> IntArray:
    """Index arrays of shape ``(n_draws, n)`` for a moving block bootstrap.

    Block starts are uniform on ``0 .. n − L``; ``⌈n/L⌉`` blocks are concatenated and
    truncated to ``n``.
    """
    if not 1 <= block_len <= n:
        raise ValueError("block_len must lie in [1, n]")
    k = math.ceil(n / block_len)
    starts = rng.integers(0, n - block_len + 1, size=(n_draws, k))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]).reshape(n_draws, -1)
    out: IntArray = idx[:, :n].astype(np.int64)
    return out


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap draws of ``δ`` and ``t_c`` for one series.

    Attributes:
        point: Point estimate on the original data.
        delta: ``(n_boot,)`` draws of ``δ``.
        t_c: ``(n_boot,)`` draws of ``t_c`` (``nan`` where ``δ* <= 0``).
        block_len: Block length ``L``.
        seed: Base seed.
        method: ``"cpu"`` (this module) or ``"torch:<device>"`` (K-5.05).
    """

    point: DeltaFit
    delta: FloatArray
    t_c: FloatArray
    block_len: int
    seed: int
    method: str = "cpu"

    @property
    def n_boot(self) -> int:
        """Number of draws."""
        return len(self.delta)

    @property
    def t_remaining(self) -> FloatArray:
        """Draws of ``t_c − t_last``."""
        out: FloatArray = self.t_c - self.point.t_last
        return out

    @property
    def frac_singular(self) -> float:
        """Share of draws with ``δ* > 0`` (a finite ``t_c``)."""
        return float(np.mean(np.isfinite(self.t_c)))

    @property
    def p_delta_positive(self) -> float:
        """One-sided bootstrap p-value for ``H0: δ <= 0``: ``(1 + #{δ* <= 0}) / (B + 1)``."""
        d = self.delta[np.isfinite(self.delta)]
        return float((1 + np.sum(d <= 0)) / (len(d) + 1))

    def interval(self, level: float = 0.9, quantity: Quantity = "delta") -> tuple[float, float]:
        """Equal-tailed percentile interval.

        Args:
            level: Coverage, e.g. 0.9.
            quantity: ``"delta"``, ``"t_c"`` or ``"t_remaining"``. Missing ``t_c``
                draws count as ``+∞``, so the upper bound may be infinite.
        """
        if not 0 < level < 1:
            raise ValueError("level must lie in (0, 1)")
        if quantity == "delta":
            x = self.delta[np.isfinite(self.delta)]
        else:
            x = self.t_c if quantity == "t_c" else self.t_remaining
            x = np.where(np.isnan(x), np.inf, x)
        if len(x) == 0:
            return (float("nan"), float("nan"))
        a = (1.0 - level) / 2.0
        big = np.finfo(np.float64).max / 4
        q = np.quantile(np.minimum(x, big), [a, 1.0 - a], method="linear")
        lo, hi = (float(v) if v < big / 2 else float("inf") for v in q)
        return (lo, hi)

    def se(self) -> float:
        """Bootstrap standard error of ``δ``."""
        return float(np.nanstd(self.delta, ddof=1))


@dataclass(frozen=True)
class _Chunk:
    t: FloatArray
    fitted: FloatArray
    resid: FloatArray
    lam: float
    edge_trim: int
    block_len: int
    n_draws: int
    seed: int
    reselect_lambda: bool


def _run_chunk(job: _Chunk) -> tuple[FloatArray, FloatArray]:
    """Worker: ``n_draws`` bootstrap refits; module level so loky can pickle it."""
    rng = np.random.default_rng(job.seed)
    n = len(job.t)
    idx = mbb_indices(n, job.block_len, job.n_draws, rng)
    ystar = job.fitted[None, :] + job.resid[idx]
    if job.reselect_lambda:
        deltas = np.empty(job.n_draws)
        tcs = np.empty(job.n_draws)
        for i in range(job.n_draws):
            try:
                f = estimate_delta(job.t, ystar[i], edge_trim=job.edge_trim)
                deltas[i], tcs[i] = f.delta, f.t_c
            except ValueError:
                deltas[i], tcs[i] = np.nan, np.nan
        return deltas, tcs
    sm = spline_smoother(job.t, job.lam)
    lx, g = sm.smooth(ystar), sm.derivative(ystar)
    a, d, _ = ols_delta(lx, g, edge_mask(n, job.edge_trim))
    tc: FloatArray = job.t[-1] + t_remaining(a, d, lx[:, -1])
    return d, tc


def _setup(
    t: FloatArray, log_y: FloatArray, lam: float | None, edge_trim: int | None, level: float
) -> tuple[DeltaFit, FloatArray, FloatArray]:
    point = estimate_delta(t, log_y, lam=lam, edge_trim=edge_trim, level=level)
    sm = spline_smoother(point.t, point.lam)
    fitted = sm.smooth(np.asarray(log_y, dtype=np.float64))
    resid = np.asarray(log_y, dtype=np.float64) - fitted
    # Smoothing shrinks residuals most where leverage is high (the ends, which drive
    # δ). Use leverage-modified residuals r_i / sqrt(1 − S_ii) (Davison and Hinkley
    # 1997, eq. 6.9), then centre.
    lev = np.clip(np.diag(sm.S), 0.0, 0.99)
    resid = resid / np.sqrt(1.0 - lev)
    return point, fitted, resid - resid.mean()


def block_bootstrap(
    t: FloatArray,
    log_y: FloatArray,
    *,
    n_boot: int = 1000,
    block_len: int | None = None,
    lam: float | None = None,
    edge_trim: int | None = None,
    reselect_lambda: bool = False,
    seed: int | None = None,
    n_jobs: int | None = None,
    draws_per_task: int = 64,
    level: float = 0.9,
) -> BootstrapResult:
    """Moving block bootstrap of ``δ`` and ``t_c`` with the smoother refit on every draw.

    Args:
        t: Strictly increasing decimal-year times.
        log_y: Observed ``ln y``.
        n_boot: Number of bootstrap draws.
        block_len: Block length; default :func:`default_block_len`.
        lam: Spline penalty; ``None`` selects it by GCV on the original data.
        edge_trim: Points dropped at each end of the regression (default 5 %).
        reselect_lambda: Re-run GCV on every draw (slower, slightly wider intervals).
        seed: Base seed; default ``Settings.seed``.
        n_jobs: Worker count for :func:`kairos.compute.pmap` (``1`` runs in-process).
        draws_per_task: Draws per pool task; results do not depend on ``n_jobs``.
        level: Coverage of the point fit's delta-method interval.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if seed is None:
        from kairos.settings import get_settings

        seed = get_settings().seed
    point, fitted, resid = _setup(t, log_y, lam, edge_trim, level)
    n = len(point.t)
    blen = default_block_len(n) if block_len is None else block_len
    sizes = [draws_per_task] * (n_boot // draws_per_task)
    if n_boot % draws_per_task:
        sizes.append(n_boot % draws_per_task)
    jobs = [
        _Chunk(
            t=point.t,
            fitted=fitted,
            resid=resid,
            lam=point.lam,
            edge_trim=point.edge_trim,
            block_len=blen,
            n_draws=size,
            seed=derive_seed(seed, "mbb", i),
            reselect_lambda=reselect_lambda,
        )
        for i, size in enumerate(sizes)
    ]
    parts = pmap(_run_chunk, jobs, n_jobs=n_jobs)
    delta = np.concatenate([p[0] for p in parts])
    t_c = np.concatenate([p[1] for p in parts])
    return BootstrapResult(point=point, delta=delta, t_c=t_c, block_len=blen, seed=seed)


def block_bootstrap_frame(
    frame: SeriesFrame,
    series_id: str | None = None,
    *,
    n_boot: int = 1000,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> BootstrapResult:
    """:func:`block_bootstrap` on one series of a ``SeriesFrame``."""
    t, log_y = frame_arrays(frame, series_id)
    return block_bootstrap(t, log_y, n_boot=n_boot, seed=seed, n_jobs=n_jobs)
