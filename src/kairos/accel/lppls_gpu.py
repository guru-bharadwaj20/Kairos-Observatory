"""Batched LPPLS grid search in ``torch`` (K-5.09).

Evaluates the profile SSE of every ``t_c × m × ω`` cell on every window in one
batched kernel: for each (window, cell) pair the three centred basis columns are
built, the 3×3 normal equations are solved with ``torch.linalg.solve`` and the SSE is
taken from the residuals directly (never from ``yᵀy − βᵀXᵀy``, which cancels badly in
float32).

Precision (CONTRIBUTING 4.2): on the GPU everything runs in **float32** with
centred and scaled inputs: ``τ`` is measured in window lengths, ``ln p`` is centred
and divided by its standard deviation per window. The grid is defined in those
normalised units, so all windows share one cell list. Pairs are processed in chunks
sized to the VRAM budget by :func:`kairos.compute.chunk_size`.

:func:`kairos.accel.lppls.lppls_sse_cells` is the float64 reference; the tests assert
both agree. Best cells are returned for float64 polishing (K-5.10).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kairos.accel.lppls import (
    DEFAULT_M_RANGE,
    DEFAULT_OMEGA_RANGE,
    DEFAULT_TC_RANGE,
    LPPLSFit,
    _make_fit,
)
from kairos.accel.synthetic import FloatArray
from kairos.compute import chunk_size, dtype_for, get_device

__all__ = ["GridResult", "lppls_grid_search", "normalised_grid"]


def normalised_grid(
    n_tc: int = 24,
    n_m: int = 12,
    n_omega: int = 12,
    *,
    tc_range: tuple[float, float] = DEFAULT_TC_RANGE,
    m_range: tuple[float, float] = DEFAULT_M_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Flattened cell list ``(frac, m, ω)`` with ``t_c = t2 + frac·(t2 − t1)``."""
    g = np.meshgrid(
        np.linspace(*tc_range, n_tc),
        np.linspace(*m_range, n_m),
        np.linspace(*omega_range, n_omega),
        indexing="ij",
    )
    return g[0].ravel(), g[1].ravel(), g[2].ravel()


@dataclass(frozen=True)
class GridResult:
    """Best grid cells per window.

    Attributes:
        windows: ``(W, 2)`` start/stop indices (``stop`` exclusive).
        tc: ``(W, k)`` critical times of the ``k`` best cells (decimal years).
        m: ``(W, k)`` exponents.
        omega: ``(W, k)`` frequencies.
        sse: ``(W, k)`` SSE in ``ln p`` units, ascending per window.
        device: Device the kernel ran on.
        sse_grid: Full ``(W, cells)`` SSE when requested, else ``None``.
    """

    windows: npt.NDArray[np.int64]
    tc: FloatArray
    m: FloatArray
    omega: FloatArray
    sse: FloatArray
    device: str
    sse_grid: FloatArray | None = None

    def best_fits(self, t: FloatArray, log_p: FloatArray) -> list[LPPLSFit]:
        """Best cell of every window as an :class:`LPPLSFit` (linear part in float64)."""
        out = []
        for w, (a, b) in enumerate(self.windows):
            out.append(_make_fit(t[a:b], log_p[a:b], self.tc[w, 0], self.m[w, 0], self.omega[w, 0]))
        return out


def lppls_grid_search(
    t: FloatArray,
    log_p: FloatArray,
    windows: Sequence[tuple[int, int]],
    *,
    n_tc: int = 24,
    n_m: int = 12,
    n_omega: int = 12,
    tc_range: tuple[float, float] = DEFAULT_TC_RANGE,
    m_range: tuple[float, float] = DEFAULT_M_RANGE,
    omega_range: tuple[float, float] = DEFAULT_OMEGA_RANGE,
    top_k: int = 1,
    device: str | None = None,
    return_grid: bool = False,
) -> GridResult:
    """Grid-search LPPLS over many windows of one series at once.

    Args:
        t: Times of the whole series (decimal years, increasing).
        log_p: ``ln p`` of the whole series.
        windows: ``(start, stop)`` index pairs, ``stop`` exclusive, each ≥ 8 points.
        n_tc: Grid points for ``(t_c − t2)/(t2 − t1)``.
        n_m: Grid points for ``m``.
        n_omega: Grid points for ``ω``.
        tc_range: Range of ``(t_c − t2)/(t2 − t1)``.
        m_range: Range of ``m``.
        omega_range: Range of ``ω``.
        top_k: Best cells kept per window.
        device: Torch device string; ``None`` uses ``Settings.device``.
        return_grid: Also return the full SSE matrix (for tests and diagnostics).
    """
    import torch

    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(log_p, dtype=np.float64)
    win = np.asarray(windows, dtype=np.int64).reshape(-1, 2)
    if len(win) == 0:
        raise ValueError("need at least one window")
    if np.any(win[:, 1] - win[:, 0] < 8) or win.min() < 0 or win[:, 1].max() > len(t):
        raise ValueError("windows must lie inside the series and hold >= 8 points")
    n_win = len(win)
    length = int((win[:, 1] - win[:, 0]).max())

    # Per-window normalised inputs, padded to the longest window (float64 on host).
    u = np.zeros((n_win, length))
    z = np.zeros((n_win, length))
    mask = np.zeros((n_win, length))
    spans = np.empty(n_win)
    scales = np.empty(n_win)
    for i, (a, b) in enumerate(win):
        tw, yw = t[a:b], y[a:b]
        span = float(tw[-1] - tw[0])
        sd = float(yw.std()) or 1.0
        k = b - a
        u[i, :k] = (tw[-1] - tw) / span
        z[i, :k] = (yw - yw.mean()) / sd
        mask[i, :k] = 1.0
        spans[i], scales[i] = span, sd

    frac, ms, oms = normalised_grid(
        n_tc, n_m, n_omega, tc_range=tc_range, m_range=m_range, omega_range=omega_range
    )
    if frac.min() <= 0:
        raise ValueError("tc_range must be > 0 (t_c after the window end)")
    n_cells = len(frac)

    dev = get_device(device)
    dt = dtype_for(dev)
    eps = 1e-6 if dt == torch.float32 else 1e-12
    u_t = torch.as_tensor(u, dtype=dt, device=dev)
    z_t = torch.as_tensor(z, dtype=dt, device=dev)
    w_t = torch.as_tensor(mask, dtype=dt, device=dev)
    nw = w_t.sum(dim=1)
    cells = torch.as_tensor(np.stack([frac, ms, oms], 1), dtype=dt, device=dev)
    sse = torch.empty((n_win, n_cells), dtype=dt, device=dev)

    n_pairs = n_win * n_cells
    itemsize = torch.finfo(dt).bits // 8
    per_chunk = chunk_size(length * itemsize * 16, n_pairs, dev)
    eye = torch.eye(3, dtype=dt, device=dev)
    with torch.no_grad():
        for p0 in range(0, n_pairs, per_chunk):
            pid = torch.arange(p0, min(p0 + per_chunk, n_pairs), device=dev)
            wi, ci = pid // n_cells, pid % n_cells
            c = cells[ci]
            w = w_t[wi]
            tau = u_t[wi] + c[:, 0:1]
            lt = torch.log(tau)
            f = torch.exp(c[:, 1:2] * lt)
            ph = c[:, 2:3] * lt
            basis = torch.stack([f, f * torch.cos(ph), f * torch.sin(ph)], dim=-1)
            basis = basis * w[..., None]
            bm = basis.sum(dim=1, keepdim=True) / nw[wi, None, None]
            bc = (basis - bm) * w[..., None]
            yc = z_t[wi]
            # Jacobi-scale the columns to unit norm: the f, f·cos, f·sin columns are
            # nearly collinear for small m, and float32 needs the conditioning help.
            norm = torch.sqrt((bc * bc).sum(dim=1)) + torch.finfo(dt).tiny
            bs = bc / norm[:, None, :]
            xtx = bs.transpose(1, 2) @ bs
            xty = (bs.transpose(1, 2) @ yc[..., None])[..., 0]
            beta = torch.linalg.solve(xtx + eps * eye, xty)
            resid = (yc - (bs @ beta[..., None])[..., 0]) * w
            sse.view(-1)[pid] = (resid * resid).sum(dim=1)
        k = min(top_k, n_cells)
        best_sse, best_idx = torch.topk(sse, k, dim=1, largest=False)

    idx = best_idx.cpu().numpy()
    scale2 = (scales**2)[:, None]
    t2 = t[win[:, 1] - 1][:, None]
    return GridResult(
        windows=win,
        tc=t2 + frac[idx] * spans[:, None],
        m=ms[idx],
        omega=oms[idx],
        sse=best_sse.to(torch.float64).cpu().numpy() * scale2,
        device=str(dev),
        sse_grid=(sse.to(torch.float64).cpu().numpy() * scale2) if return_grid else None,
    )
