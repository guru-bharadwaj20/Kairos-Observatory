"""Float64 polishing of the best GPU grid cells (K-5.10).

The float32 grid (:mod:`kairos.accel.lppls_gpu`) is a screen. Its ``top_k`` cells per
window are polished on the CPU pool with the float64 Nelder–Mead of
:func:`kairos.accel.lppls.polish`, and the lowest-SSE result per window is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kairos.accel.lppls import (
    DEFAULT_M_RANGE,
    DEFAULT_OMEGA_RANGE,
    DEFAULT_TC_RANGE,
    LPPLSFit,
    polish,
)
from kairos.accel.lppls_gpu import GridResult
from kairos.accel.synthetic import FloatArray
from kairos.compute import pmap

__all__ = ["refine_grid"]


@dataclass(frozen=True)
class _Job:
    window: int
    t: FloatArray
    y: FloatArray
    start: tuple[float, float, float]
    tc_range: tuple[float, float]
    m_range: tuple[float, float]
    omega_range: tuple[float, float]


def _polish_job(job: _Job) -> tuple[int, LPPLSFit]:
    """Worker: polish one start cell; module level so loky can pickle it."""
    fit = polish(
        job.t,
        job.y,
        job.start,
        tc_range=job.tc_range,
        m_range=job.m_range,
        omega_range=job.omega_range,
    )
    return job.window, fit


def refine_grid(
    t: FloatArray,
    log_p: FloatArray,
    grid: GridResult,
    *,
    tc_range: tuple[float, float] = DEFAULT_TC_RANGE,
    m_range: tuple[float, float] = DEFAULT_M_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
    n_jobs: int | None = None,
) -> list[LPPLSFit]:
    """Polish every kept grid cell in float64 and return the best fit per window.

    Args:
        t: Times of the whole series.
        log_p: ``ln p`` of the whole series.
        grid: Output of :func:`kairos.accel.lppls_gpu.lppls_grid_search`.
        tc_range: Search bounds used by the polisher (as in the grid).
        m_range: Bounds on ``m``.
        omega_range: Bounds on ``ω``.
        n_jobs: Worker count for :func:`kairos.compute.pmap`.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(log_p, dtype=np.float64)
    jobs = [
        _Job(
            window=w,
            t=t[a:b],
            y=y[a:b],
            start=(float(grid.tc[w, j]), float(grid.m[w, j]), float(grid.omega[w, j])),
            tc_range=tc_range,
            m_range=m_range,
            omega_range=omega_range,
        )
        for w, (a, b) in enumerate(grid.windows)
        for j in range(grid.tc.shape[1])
    ]
    best: dict[int, LPPLSFit] = {}
    for w, fit in pmap(_polish_job, jobs, n_jobs=n_jobs):
        if w not in best or fit.sse < best[w].sse:
            best[w] = fit
    return [best[w] for w in range(len(grid.windows))]
