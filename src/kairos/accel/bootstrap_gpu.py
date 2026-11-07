"""Vectorised moving block bootstrap in ``torch`` (K-5.05).

Same estimator as :mod:`kairos.accel.bootstrap` (fixed-``λ`` spline refit per draw),
but all draws of a chunk are computed at once: for fixed ``λ`` the spline is linear,
so with ``ln y* = ln x̂ + r*``

    ln x* = S·ln x̂ + S·r*,    g* = D·ln x̂ + D·r*

and the ``ln g`` on ``ln x`` regression is a closed-form weighted OLS over a batch axis.
The deterministic parts ``S·ln x̂`` and ``D·ln x̂`` are computed once in float64 and
centred, so the float32 GPU kernel only handles residual-sized numbers
(CONTRIBUTING 4.2). Chunks are sized to the VRAM budget with
:func:`kairos.compute.chunk_size`.

The draws are statistically, not bitwise, equivalent to the CPU reference (different
random streams); the test suite checks this with a two-sample KS test.
"""

from __future__ import annotations

import numpy as np

from kairos.accel.bootstrap import BootstrapResult, _setup, default_block_len
from kairos.accel.growth import edge_mask, spline_smoother
from kairos.accel.synthetic import FloatArray
from kairos.compute import chunk_size, derive_seed, dtype_for, get_device

__all__ = ["block_bootstrap_torch"]


def block_bootstrap_torch(
    t: FloatArray,
    log_y: FloatArray,
    *,
    n_boot: int = 100_000,
    block_len: int | None = None,
    lam: float | None = None,
    edge_trim: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    max_chunk: int = 65_536,
    level: float = 0.9,
) -> BootstrapResult:
    """Moving block bootstrap of ``δ`` and ``t_c``, batched on GPU (or CPU) with torch.

    Args:
        t: Strictly increasing decimal-year times.
        log_y: Observed ``ln y``.
        n_boot: Number of draws (10⁵ takes well under a second on the workstation GPU).
        block_len: Block length; default :func:`kairos.accel.bootstrap.default_block_len`.
        lam: Spline penalty; ``None`` selects it by GCV on the original data.
        edge_trim: Points dropped at each end of the regression (default 5 %).
        seed: Base seed; default ``Settings.seed``.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``...; ``None`` uses ``Settings.device``.
            Precision follows :func:`kairos.compute.dtype_for` (float32 on GPU).
        max_chunk: Upper bound on draws per kernel launch.
        level: Coverage of the point fit's delta-method interval.
    """
    import torch

    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if seed is None:
        from kairos.settings import get_settings

        seed = get_settings().seed
    point, fitted, resid = _setup(t, log_y, lam, edge_trim, level)
    n = len(point.t)
    blen = default_block_len(n) if block_len is None else block_len
    if not 1 <= blen <= n:
        raise ValueError("block_len must lie in [1, n]")
    n_blocks = -(-n // blen)

    dev = get_device(device)
    dt = dtype_for(dev)
    sm = spline_smoother(point.t, point.lam)
    lx_hat64 = sm.smooth(fitted)
    centre = float(lx_hat64.mean())
    lx_hat = torch.as_tensor(lx_hat64 - centre, dtype=dt, device=dev)
    g_hat = torch.as_tensor(sm.derivative(fitted), dtype=dt, device=dev)
    s_t = torch.as_tensor(sm.S.T.copy(), dtype=dt, device=dev)
    d_t = torch.as_tensor(sm.D.T.copy(), dtype=dt, device=dev)
    r_t = torch.as_tensor(resid, dtype=dt, device=dev)
    base_w = torch.as_tensor(edge_mask(n, point.edge_trim), dtype=dt, device=dev)
    offsets = torch.arange(blen, device=dev)

    gen = torch.Generator(device=dev)
    gen.manual_seed(derive_seed(seed, "mbb-torch"))
    bytes_per_draw = n * torch.finfo(dt).bits // 8 * 10
    per_chunk = min(max_chunk, chunk_size(bytes_per_draw, n_boot, dev))

    deltas: list[FloatArray] = []
    tcs: list[FloatArray] = []
    done = 0
    with torch.no_grad():
        while done < n_boot:
            b = min(per_chunk, n_boot - done)
            starts = torch.randint(0, n - blen + 1, (b, n_blocks), generator=gen, device=dev)
            idx = (starts[:, :, None] + offsets).reshape(b, -1)[:, :n]
            rs = r_t[idx]
            lx = lx_hat + rs @ s_t
            g = g_hat + rs @ d_t
            pos = g > 0
            w = torch.where(pos, base_w, torch.zeros_like(base_w))
            lg = torch.log(torch.where(pos, g, torch.ones_like(g)))
            sw = w.sum(dim=1)
            mx = (w * lx).sum(dim=1) / sw
            my = (w * lg).sum(dim=1) / sw
            dx = lx - mx[:, None]
            sxx = (w * dx * dx).sum(dim=1)
            sxy = (w * dx * (lg - my[:, None])).sum(dim=1)
            d = sxy / sxx
            a = my - d * mx
            d = torch.where(sw >= 3, d, torch.full_like(d, float("nan")))
            # T = 1/(δ g_last), g_last = exp(a + δ ln x_last) in centred coordinates.
            t_rem = 1.0 / (d * torch.exp(a + d * lx[:, -1]))
            t_rem = torch.where(d > 0, t_rem, torch.full_like(t_rem, float("nan")))
            d64 = d.to(torch.float64).cpu().numpy()
            deltas.append(d64)
            tcs.append(point.t_last + t_rem.to(torch.float64).cpu().numpy())
            done += b

    return BootstrapResult(
        point=point,
        delta=np.concatenate(deltas),
        t_c=np.concatenate(tcs),
        block_len=blen,
        seed=seed,
        method=f"torch:{dev}",
    )
