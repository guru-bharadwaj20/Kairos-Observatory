"""Coverage check of the bootstrap intervals for ``δ`` (K-5.06).

Simulates many hyperbolic series with a known ``δ``, runs the CPU block bootstrap
(:func:`kairos.accel.bootstrap.block_bootstrap`) on each, and reports the share of
nominal-``level`` percentile intervals that contain the truth. Trials run in parallel
with :func:`kairos.compute.pmap`; each trial has its own derived seed, so the result
is reproducible and independent of the worker count.

A well-calibrated 90 % interval covers the truth in 85–95 % of trials. Measured on
this workstation (400 trials, 400 draws, 240 monthly points, iid log noise 0.1):
``δ = 1.0`` → 0.87; ``δ = 0.5`` → 0.84; log noise 0.05 → 0.74; AR(1) = 0.3 → 0.68.
The intervals are slightly narrow because the bootstrap cannot reproduce the
sample-to-sample variation of the GCV-chosen ``λ``, and autocorrelated noise makes
GCV undersmooth. Treat ``δ`` intervals on strongly autocorrelated series as optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kairos.accel.bootstrap import block_bootstrap
from kairos.accel.synthetic import FloatArray, hyperbolic_series
from kairos.compute import derive_seed, pmap

__all__ = ["CoverageResult", "coverage_check"]


@dataclass(frozen=True)
class _Trial:
    seed: int
    n: int
    delta: float
    t_c: float
    noise: float
    ar: float
    n_boot: int
    level: float
    reselect_lambda: bool = False


def _run_trial(job: _Trial) -> tuple[float, float, float]:
    """Worker: one simulated series → ``(δ̂, lo, hi)``."""
    s = hyperbolic_series(
        job.n, delta=job.delta, t_c=job.t_c, noise=job.noise, ar=job.ar, seed=job.seed
    )
    res = block_bootstrap(
        s.t,
        s.log_y,
        n_boot=job.n_boot,
        seed=derive_seed(job.seed, "boot"),
        n_jobs=1,
        reselect_lambda=job.reselect_lambda,
    )
    lo, hi = res.interval(job.level)
    return res.point.delta, lo, hi


@dataclass(frozen=True)
class CoverageResult:
    """Outcome of :func:`coverage_check`.

    Attributes:
        coverage: Share of intervals containing the true ``δ``.
        level: Nominal level.
        n_trials: Number of simulated series.
        delta_true: True ``δ``.
        estimates: Point estimates per trial.
        lo: Lower interval bounds per trial.
        hi: Upper interval bounds per trial.
    """

    coverage: float
    level: float
    n_trials: int
    delta_true: float
    estimates: FloatArray
    lo: FloatArray
    hi: FloatArray

    @property
    def mc_se(self) -> float:
        """Monte Carlo standard error of :attr:`coverage`."""
        return float(np.sqrt(self.coverage * (1 - self.coverage) / self.n_trials))

    def calibrated(self, tol: float = 0.05) -> bool:
        """Whether coverage lies within ``level ± tol`` (default 85–95 % for 90 %)."""
        return abs(self.coverage - self.level) <= tol


def coverage_check(
    n_trials: int = 200,
    *,
    level: float = 0.9,
    delta: float = 0.5,
    t_c: float = 2022.0,
    n: int = 240,
    noise: float = 0.1,
    ar: float = 0.0,
    n_boot: int = 400,
    seed: int = 0,
    n_jobs: int | None = None,
    reselect_lambda: bool = False,
) -> CoverageResult:
    """Share of bootstrap ``level`` intervals that cover the true ``δ``.

    Args:
        n_trials: Simulated series.
        level: Interval level.
        delta: True ``δ``.
        t_c: True critical time (decimal years); must lie after the last point.
        n: Monthly points per series (start 2000-01).
        noise: Marginal sd of the log noise.
        ar: AR(1) coefficient of the log noise.
        n_boot: Bootstrap draws per trial.
        seed: Base seed.
        n_jobs: Worker count for :func:`kairos.compute.pmap`.
        reselect_lambda: Re-select ``λ`` by GCV on every bootstrap draw.
    """
    jobs = [
        _Trial(
            derive_seed(seed, "coverage", i),
            n,
            delta,
            t_c,
            noise,
            ar,
            n_boot,
            level,
            reselect_lambda,
        )
        for i in range(n_trials)
    ]
    out = np.asarray(pmap(_run_trial, jobs, n_jobs=n_jobs), dtype=np.float64)
    est, lo, hi = out[:, 0], out[:, 1], out[:, 2]
    covered = (lo <= delta) & (delta <= hi)
    return CoverageResult(
        coverage=float(covered.mean()),
        level=level,
        n_trials=n_trials,
        delta_true=delta,
        estimates=est,
        lo=lo,
        hi=hi,
    )
