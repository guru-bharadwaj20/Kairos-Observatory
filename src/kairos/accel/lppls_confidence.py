"""LPPLS confidence indicator (K-5.11).

Sornette-style "DS LPPLS confidence": at each end date ``t2`` fit LPPLS on a family
of shrinking windows ``[t2 − L, t2]`` and report the share of windows whose fit
passes the qualification filter (:mod:`kairos.accel.lppls_filter`). A value near 1
means the bubble signature is robust to the choice of window start.

All windows of all end dates go through a single batched grid search
(:func:`kairos.accel.lppls_gpu.lppls_grid_search`, GPU float32 when available); the
linear parameters of each window's best cell are recomputed in float64 before the
filter is applied. ``refine=True`` additionally polishes every window in float64
(:mod:`kairos.accel.lppls_refine`), which is slower but sharper.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from kairos.accel.growth import frame_arrays
from kairos.accel.lppls_filter import DEFAULT_FILTER, LPPLSFilter, qualify
from kairos.accel.lppls_gpu import lppls_grid_search
from kairos.accel.lppls_refine import refine_grid
from kairos.accel.synthetic import FloatArray
from kairos.types import SeriesFrame

__all__ = ["lppls_confidence", "lppls_confidence_frame"]


def lppls_confidence(
    t: FloatArray,
    log_p: FloatArray,
    *,
    ends: Sequence[int] | None = None,
    window_lengths: Sequence[int] = tuple(range(36, 121, 6)),
    flt: LPPLSFilter = DEFAULT_FILTER,
    n_tc: int = 16,
    n_m: int = 10,
    n_omega: int = 12,
    device: str | None = None,
    refine: bool = False,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """LPPLS confidence for each end index.

    Args:
        t: Times (decimal years, increasing).
        log_p: ``ln p``.
        ends: End indices (inclusive) to evaluate; default every index that admits
            the shortest window.
        window_lengths: Window lengths in points (monthly data: months).
        flt: Qualification filter.
        n_tc: Grid points for ``t_c``.
        n_m: Grid points for ``m``.
        n_omega: Grid points for ``ω``.
        device: Torch device for the grid search.
        refine: Polish each window in float64 before filtering.
        n_jobs: Workers for the refinement pool.

    Returns:
        Frame with columns ``t``, ``end``, ``n_windows``, ``n_qualified`` and
        ``confidence`` (``n_qualified / n_windows``), one row per end index.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(log_p, dtype=np.float64)
    lengths = sorted({int(v) for v in window_lengths if v >= 8})
    if not lengths:
        raise ValueError("need window lengths of at least 8 points")
    if ends is None:
        ends = range(lengths[0] - 1, len(t))
    windows: list[tuple[int, int]] = []
    owner: list[int] = []
    end_list = [int(e) for e in ends]
    for i, e in enumerate(end_list):
        for length in lengths:
            if e + 1 - length >= 0:
                windows.append((e + 1 - length, e + 1))
                owner.append(i)
    n_win = np.zeros(len(end_list), dtype=np.int64)
    n_ok = np.zeros(len(end_list), dtype=np.int64)
    if windows:
        grid = lppls_grid_search(t, y, windows, n_tc=n_tc, n_m=n_m, n_omega=n_omega, device=device)
        fits = refine_grid(t, y, grid, n_jobs=n_jobs) if refine else grid.best_fits(t, y)
        for i, fit in zip(owner, fits, strict=True):
            n_win[i] += 1
            n_ok[i] += qualify(fit, flt).qualified
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(n_win > 0, n_ok / np.maximum(n_win, 1), np.nan)
    return pd.DataFrame(
        {
            "t": t[end_list] if end_list else np.array([]),
            "end": end_list,
            "n_windows": n_win,
            "n_qualified": n_ok,
            "confidence": conf,
        }
    )


def lppls_confidence_frame(
    frame: SeriesFrame,
    series_id: str | None = None,
    *,
    last: int | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """:func:`lppls_confidence` on one ``SeriesFrame`` series, with a ``ds`` column.

    Args:
        frame: Long-format frame.
        series_id: Series to use (optional when the frame holds one).
        last: Only evaluate the last ``last`` end dates.
        **kwargs: Passed to :func:`lppls_confidence`.
    """
    t, log_p = frame_arrays(frame, series_id)
    sub = frame if series_id is None else frame[frame["series_id"] == series_id]
    ds = sub.dropna(subset=["y"]).sort_values("ds")["ds"].to_numpy()
    if last is not None:
        kwargs.setdefault("ends", range(max(0, len(t) - last), len(t)))
    out = lppls_confidence(t, log_p, **kwargs)  # type: ignore[arg-type]
    out.insert(0, "ds", ds[out["end"].to_numpy()])
    return out
