"""Prior sensitivity sweep over ``changepoint_prior_scale`` x ``n_changepoints``.

Conclusions drawn from changepoints must not hinge on one prior setting. The
sweep refits each series on a grid and records, per cell:

* ``final_rate``: growth rate at the end of the history (ln units per year);
* ``n_significant``: changepoints with ``|delta| >= min_magnitude``;
* ``yhat_end``: log forecast at the end of the horizon.

A cell is **stable** when ``metric`` differs from every grid neighbour (up,
down, left, right) by at most ``tol``. The *stable region* is the largest
4-connected set of stable cells; the recommended setting is the stable cell
closest to the configured default on the log-``changepoint_prior_scale`` x
``n_changepoints`` grid.

Fits run through :func:`kairos.compute.pmap`, one task per series x
``changepoint_prior_scale`` row (several fits per task, as the benchmark advises).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import ndimage

from kairos.compute.parallel import pmap, resolve_n_jobs
from kairos.models.changepoints import DEFAULT_MIN_MAGNITUDE
from kairos.models.null_model import ProphetConfig, fit_null_model, select_series
from kairos.types import SeriesFrame

__all__ = [
    "DEFAULT_N_CHANGEPOINTS",
    "DEFAULT_PRIOR_SCALES",
    "SensitivityReport",
    "prior_sensitivity",
    "stability",
]

DEFAULT_PRIOR_SCALES: Final = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_N_CHANGEPOINTS: Final = (10, 15, 25, 35, 50)
METRICS: Final = ("final_rate", "n_significant", "yhat_end")


@dataclass(frozen=True)
class _Row:
    frame: pd.DataFrame
    config: ProphetConfig
    prior_scale: float
    n_changepoints: tuple[int, ...]
    seed: int | None
    horizon: int
    min_magnitude: float


def _fit_row(task: _Row) -> list[dict[str, Any]]:
    """Fit one series for one prior scale across every ``n_changepoints`` value."""
    out = []
    for n_cp in task.n_changepoints:
        cfg = task.config.model_copy(
            update={"changepoint_prior_scale": task.prior_scale, "n_changepoints": n_cp}
        )
        fit = fit_null_model(
            task.frame, config=cfg, seed=task.seed, horizon=task.horizon, keep_model=False
        )
        out.append(
            {
                "series_id": fit.series_id,
                "changepoint_prior_scale": task.prior_scale,
                "n_changepoints": n_cp,
                "final_rate": fit.final_rate,
                "n_significant": int((np.abs(fit.deltas) >= task.min_magnitude).sum()),
                "yhat_end": float(fit.forecast["yhat_log"].iloc[-1]),
                "fit_seconds": fit.fit_seconds,
            }
        )
    return out


def stability(values: np.ndarray[Any, Any], tol: float) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Mask of grid cells whose value is within ``tol`` of every 4-neighbour."""
    v = np.asarray(values, dtype="float64")
    worst = np.zeros_like(v)
    worst[1:, :] = np.maximum(worst[1:, :], np.abs(v[1:, :] - v[:-1, :]))
    worst[:-1, :] = np.maximum(worst[:-1, :], np.abs(v[1:, :] - v[:-1, :]))
    worst[:, 1:] = np.maximum(worst[:, 1:], np.abs(v[:, 1:] - v[:, :-1]))
    worst[:, :-1] = np.maximum(worst[:, :-1], np.abs(v[:, 1:] - v[:, :-1]))
    return np.asarray(np.isfinite(v) & (worst <= tol), dtype=bool)


@dataclass(frozen=True)
class SensitivityReport:
    """Sweep results for one or more series.

    Attributes:
        grid: One row per series x cell with the metrics, ``stable`` and ``in_region``.
        regions: One row per series: region size, share of grid, metric range inside
            the region and the recommended setting.
        metric: Metric used for stability.
        tol: Tolerance used.
        wall_seconds: Wall time of the sweep.
    """

    grid: pd.DataFrame
    regions: pd.DataFrame
    metric: str
    tol: float
    wall_seconds: float

    def to_markdown(self) -> str:
        """Short stable-region report in Markdown."""
        lines = [
            f"# Prior sensitivity ({self.metric}, tol={self.tol:g})",
            "",
            "| series | stable cells | share | metric range | recommended cps | recommended n_cp |",
            "|---|--:|--:|---|--:|--:|",
        ]
        for r in self.regions.itertuples(index=False):
            lines.append(
                f"| {r.series_id} | {r.region_cells} | {r.region_share:.0%} | "
                f"{r.metric_min:.3g} .. {r.metric_max:.3g} | "
                f"{r.recommended_prior_scale:g} | {r.recommended_n_changepoints} |"
            )
        return "\n".join(lines) + "\n"


def _region(
    sub: pd.DataFrame,
    scales: Sequence[float],
    ncps: Sequence[int],
    metric: str,
    tol: float,
    default: tuple[float, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    table = sub.pivot_table(
        index="changepoint_prior_scale", columns="n_changepoints", values=metric, aggfunc="first"
    )
    table = table.reindex(index=list(scales), columns=list(ncps))
    values = table.to_numpy(dtype="float64")
    stable = stability(values, tol)
    labels, n = ndimage.label(stable)
    if n:
        sizes = ndimage.sum(stable, labels, index=range(1, n + 1))
        region = labels == (int(np.argmax(sizes)) + 1)
    else:
        region = np.zeros_like(stable)
    # Distance on the (log scale index, n_cp index) grid from the default setting.
    log_s = np.log10(np.asarray(scales, dtype="float64"))
    s_step = float(np.median(np.diff(log_s))) if len(log_s) > 1 else 1.0
    c_arr = np.asarray(ncps, dtype="float64")
    c_step = float(np.median(np.diff(c_arr))) if len(c_arr) > 1 else 1.0
    ds = (log_s[:, None] - math.log10(default[0])) / s_step
    dc = (c_arr[None, :] - default[1]) / c_step
    dist = np.where(region, np.hypot(ds, dc), np.inf)
    rec: tuple[float, int] | tuple[float, None] = (float("nan"), None)
    if np.isfinite(dist).any():
        i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        rec = (float(scales[int(i)]), int(ncps[int(j)]))
    flags = pd.DataFrame(
        {
            "changepoint_prior_scale": np.repeat(list(scales), len(ncps)),
            "n_changepoints": np.tile(list(ncps), len(scales)),
            "stable": stable.reshape(-1),
            "in_region": region.reshape(-1),
        }
    )
    inside = values[region]
    summary = {
        "region_cells": int(region.sum()),
        "region_share": float(region.mean()),
        "metric_min": float(inside.min()) if inside.size else float("nan"),
        "metric_max": float(inside.max()) if inside.size else float("nan"),
        "recommended_prior_scale": rec[0],
        "recommended_n_changepoints": rec[1],
    }
    return flags, summary


def prior_sensitivity(
    frame: SeriesFrame,
    *,
    series_ids: Sequence[str] | None = None,
    prior_scales: Sequence[float] = DEFAULT_PRIOR_SCALES,
    n_changepoints: Sequence[int] = DEFAULT_N_CHANGEPOINTS,
    metric: str = "final_rate",
    tol: float = 0.05,
    config: ProphetConfig | None = None,
    seed: int | None = None,
    horizon: int = 12,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    n_jobs: int | None = None,
    desc: str | None = None,
) -> SensitivityReport:
    """Refit every series on the prior grid and report the stable region.

    Args:
        frame: ``SeriesFrame`` in raw units.
        series_ids: Subset of series (default: all).
        prior_scales: ``changepoint_prior_scale`` values, ascending.
        n_changepoints: ``n_changepoints`` values, ascending.
        metric: One of ``final_rate``, ``n_significant``, ``yhat_end``.
        tol: Largest neighbour difference of ``metric`` for a stable cell.
        config: Base settings; the default setting anchors the recommendation.
        seed: Master seed (each series keeps its own derived seed in every cell).
        horizon: Forecast months for ``yhat_end``.
        min_magnitude: Threshold for ``n_significant``.
        n_jobs: Pool size.
        desc: Progress-bar label.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    config = config or ProphetConfig()
    ids = (
        list(series_ids)
        if series_ids is not None
        else [str(s) for s in pd.unique(frame["series_id"])]
    )
    frames = [select_series(frame, sid)[1] for sid in ids]
    tasks = [
        _Row(
            f, config, float(s), tuple(int(n) for n in n_changepoints), seed, horizon, min_magnitude
        )
        for f in frames
        for s in prior_scales
    ]
    start = time.perf_counter()
    rows = pmap(_fit_row, tasks, n_jobs=resolve_n_jobs(n_jobs, len(tasks)), desc=desc)
    wall = time.perf_counter() - start
    grid = pd.DataFrame([r for batch in rows for r in batch])

    default = (config.changepoint_prior_scale, config.n_changepoints)
    flag_parts, summaries = [], []
    for sid in ids:
        sub = grid.loc[grid["series_id"] == sid]
        flags, summary = _region(sub, prior_scales, n_changepoints, metric, tol, default)
        flag_parts.append(flags.assign(series_id=sid))
        summaries.append({"series_id": sid, **summary})
    grid = grid.merge(
        pd.concat(flag_parts, ignore_index=True),
        on=["series_id", "changepoint_prior_scale", "n_changepoints"],
    )
    return SensitivityReport(grid, pd.DataFrame(summaries), metric, tol, wall)
