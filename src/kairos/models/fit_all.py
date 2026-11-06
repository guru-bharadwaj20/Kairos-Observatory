"""Fit the null model to every series through :func:`kairos.compute.pmap`.

A single Prophet fit takes about 0.1 s, so the per-task overhead of the process
pool (pickling, CmdStan start-up) is not negligible. Following
``docs/benchmarks.md``, several series are fitted per pool task; the default
batch puts about four tasks on every worker so the dynamic scheduler can still
balance P- and E-cores.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from kairos.compute.cache import cached_fit
from kairos.compute.parallel import pmap, resolve_n_jobs
from kairos.logging import get_logger
from kairos.models.null_model import NullModelFit, ProphetConfig, fit_null_model
from kairos.types import SeriesFrame, validate_series_frame

__all__ = ["FitAllResult", "default_series_per_task", "fit_all", "forecast_frame"]

_log = get_logger("models.fit_all")


@dataclass(frozen=True)
class _Task:
    """One pool task: a batch of single-series frames sharing settings."""

    frames: tuple[pd.DataFrame, ...]
    config: ProphetConfig
    seed: int | None
    horizon: int | None
    cache: bool


def _fit_one(
    frame: pd.DataFrame, config: ProphetConfig, seed: int | None, horizon: int | None
) -> NullModelFit:
    return fit_null_model(frame, config=config, seed=seed, horizon=horizon, keep_model=False)


_fit_one_cached = cached_fit(version="1")(_fit_one)


def _fit_batch(task: _Task) -> list[NullModelFit]:
    """Fit every series in ``task`` (module level, so ``loky`` can pickle it)."""
    fit = _fit_one_cached if task.cache else _fit_one
    return [fit(frame, task.config, task.seed, task.horizon) for frame in task.frames]


def default_series_per_task(n_series: int, n_workers: int, tasks_per_worker: int = 4) -> int:
    """Series per pool task so each worker gets about ``tasks_per_worker`` tasks."""
    return max(1, math.ceil(n_series / max(1, n_workers * tasks_per_worker)))


@dataclass
class FitAllResult:
    """Null-model fits of every series plus timing.

    Attributes:
        fits: Fit per ``series_id`` (without Prophet objects).
        wall_seconds: Wall time of the whole parallel fit.
        n_jobs: Workers used.
        series_per_task: Series fitted per pool task.
    """

    fits: dict[str, NullModelFit]
    wall_seconds: float
    n_jobs: int
    series_per_task: int
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def fit_seconds_total(self) -> float:
        """Sum of per-series fit times (CPU-side work)."""
        return sum(f.fit_seconds for f in self.fits.values())

    def summary(self) -> pd.DataFrame:
        """One row per series: frontier flag, size, rates and fit time."""
        rows = [
            {
                "series_id": f.series_id,
                "frontier": f.frontier,
                "n_obs": int(f.history["y"].notna().sum()),
                "start": f.history["ds"].min(),
                "end": f.history["ds"].max(),
                "base_rate": f.base_rate,
                "final_rate": f.final_rate,
                "fit_seconds": f.fit_seconds,
            }
            for f in self.fits.values()
        ]
        return pd.DataFrame(rows)


def forecast_frame(fits: dict[str, NullModelFit]) -> pd.DataFrame:
    """Stack every forecast into one long frame with a ``series_id`` column."""
    parts = [f.forecast.assign(series_id=sid) for sid, f in fits.items()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fit_all(
    frame: SeriesFrame,
    *,
    series_ids: Sequence[str] | None = None,
    config: ProphetConfig | None = None,
    seed: int | None = None,
    horizon: int | None = None,
    n_jobs: int | None = None,
    series_per_task: int | None = None,
    cache: bool = False,
    desc: str | None = None,
) -> FitAllResult:
    """Fit the null model to every series of ``frame`` in parallel.

    Args:
        frame: Long ``SeriesFrame`` in raw units.
        series_ids: Subset to fit (default: all, in order of first appearance).
        config: Model settings shared by every series.
        seed: Master seed; each series derives its own CmdStan seed.
        horizon: Forecast months past each history.
        n_jobs: Pool size (default ``Settings.n_jobs``); ``1`` fits in-process.
        series_per_task: Series per pool task (default :func:`default_series_per_task`).
        cache: Load unchanged fits from the on-disk fit cache.
        desc: Progress-bar label.

    Returns:
        A :class:`FitAllResult`; ``fits`` keep the input order.
    """
    validate_series_frame(frame)
    config = config or ProphetConfig()
    ids = (
        list(series_ids)
        if series_ids is not None
        else [str(s) for s in pd.unique(frame["series_id"])]
    )
    workers = resolve_n_jobs(n_jobs, len(ids))
    per_task = series_per_task or default_series_per_task(len(ids), workers)
    groups = {str(k): g for k, g in frame.groupby("series_id", sort=False)}
    missing = [s for s in ids if s not in groups]
    if missing:
        raise ValueError(f"series not in frame: {missing}")

    tasks = [
        _Task(tuple(groups[s] for s in ids[i : i + per_task]), config, seed, horizon, cache)
        for i in range(0, len(ids), per_task)
    ]
    start = time.perf_counter()
    batches = pmap(_fit_batch, tasks, n_jobs=workers, desc=desc)
    wall = time.perf_counter() - start

    fits = {f.series_id: f for batch in batches for f in batch}
    _log.info(
        "fitted %d series in %.2f s (%d workers, %d per task)", len(fits), wall, workers, per_task
    )
    return FitAllResult(fits, wall, workers, per_task, {"n_tasks": len(tasks)})
