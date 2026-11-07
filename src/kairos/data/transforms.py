"""Small numeric transforms shared by the loaders and the composite indices."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from kairos.data.schema import month_start

__all__ = ["EPS", "annual_to_monthly", "inv_logit", "logit", "transform_values"]

EPS = 1e-6
"""Clip bounded scores to ``[EPS, 1 - EPS]`` before the logit."""


def logit(p: Any, eps: float = EPS) -> npt.NDArray[np.float64]:
    """Return ``ln(p / (1 - p))`` with ``p`` clipped to ``[eps, 1 - eps]``.

    Bounded scores approach 1 ever more slowly; on the logit scale steady progress
    stays linear, so saturation does not look like a slowdown.
    """
    q = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.asarray(np.log(q / (1 - q)), dtype=float)


def inv_logit(x: Any) -> npt.NDArray[np.float64]:
    """Inverse of :func:`logit`."""
    return np.asarray(1 / (1 + np.exp(-np.asarray(x, dtype=float))), dtype=float)


def transform_values(y: Any, transform: str) -> npt.NDArray[np.float64]:
    """Apply a :data:`kairos.types.Transform` (``log``, ``logit`` or ``none``)."""
    arr = np.asarray(y, dtype=float)
    if transform == "log":
        return np.asarray(np.log(np.clip(arr, np.finfo(float).tiny, None)), dtype=float)
    if transform == "logit":
        return logit(arr)
    if transform == "none":
        return arr
    raise ValueError(f"unknown transform {transform!r}")


def annual_to_monthly(
    ds: Any, y: Any, *, method: str = "loglinear", month: int = 7
) -> pd.DataFrame:
    """Interpolate sparse (e.g. annual) points to a monthly grid.

    Args:
        ds: Dates, or integer years (a bare year is placed in ``month``).
        y: Values (strictly positive for ``loglinear``).
        method: ``"loglinear"`` or ``"linear"``.
        month: Month a bare year is assigned to (7 = mid-year).

    Returns:
        Monthly frame ``ds``, ``y``, ``interpolated`` spanning first to last point.
    """
    s = pd.Series(ds)
    if pd.api.types.is_integer_dtype(s):
        s = pd.to_datetime(pd.DataFrame({"year": s, "month": month, "day": 1}))
    pts = pd.DataFrame({"ds": month_start(s).to_numpy(), "y": np.asarray(y, dtype=float)})
    pts = pts.groupby("ds", as_index=False)["y"].mean().sort_values("ds")
    months = pd.date_range(pts["ds"].iloc[0], pts["ds"].iloc[-1], freq="MS")
    x = months.get_indexer(pd.DatetimeIndex(pts["ds"]))
    grid = np.arange(len(months))
    if method == "loglinear":
        vals = np.exp(np.interp(grid, x, np.log(pts["y"].to_numpy())))
    elif method == "linear":
        vals = np.interp(grid, x, pts["y"].to_numpy())
    else:
        raise ValueError(f"unknown method {method!r}")
    return pd.DataFrame(
        {"ds": months.astype("datetime64[ns]"), "y": vals, "interpolated": ~np.isin(grid, x)}
    )
