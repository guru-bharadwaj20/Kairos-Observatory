"""Gap velocity ``dG/dt`` with uncertainty (K-7.02).

``dG/dt`` at month ``t`` is the slope of an ordinary least-squares line through the
last ``window`` months of ``G`` (trailing, so a reading never uses later data),
expressed in log units **per year**: ``dG/dt = 0.3`` means capability is outgrowing
human capacity by about e^0.3 − 1 ≈ 35% a year.

Uncertainty comes from two sources, reported separately and combined conservatively:

* **Bootstrap over index construction.** Each replicate redraws the components of
  each side with replacement (so the composite's make-up is uncertain) and, where a
  component has ``y_lo``/``y_hi``, perturbs every observation by log-normal noise
  whose 90% interval matches that band. Replicates run on :func:`kairos.compute.pmap`
  with seeds from :func:`kairos.compute.derive_seed`, so results do not depend on the
  worker count.
* **Leave-one-out.** ``dG/dt`` is recomputed with each component dropped in turn
  (sides with a single component are skipped).

The reported interval ``[dgdt_lo, dgdt_hi]`` is the union of the bootstrap
percentile interval and the leave-one-out range.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from kairos.compute import derive_seed, pmap
from kairos.jonas.gap import jonas_gap, side_components
from kairos.logging import get_logger
from kairos.settings import get_settings
from kairos.types import SeriesFrame

__all__ = [
    "BAND_COVERAGE",
    "VELOCITY_COLUMNS",
    "decimal_years",
    "gap_slope",
    "gap_velocity",
    "leave_one_out",
]

log = get_logger(__name__)

BAND_COVERAGE: Final = 0.90
"""Coverage assumed for ``y_lo``/``y_hi`` bands when perturbing observations."""

VELOCITY_COLUMNS: Final = (
    "ds",
    "gap",
    "dgdt",
    "boot_lo",
    "boot_hi",
    "boot_se",
    "loo_min",
    "loo_max",
    "dgdt_lo",
    "dgdt_hi",
)


def decimal_years(ds: pd.Series | pd.DatetimeIndex) -> NDArray[np.float64]:
    """Return month-start dates as decimal years (``2020-07-01`` → ``2020.5``)."""
    idx = pd.DatetimeIndex(ds)
    return np.asarray(idx.year + (idx.month - 1) / 12.0, dtype="float64")


def gap_slope(gap: pd.DataFrame, window: int = 12, column: str = "gap") -> pd.Series:
    """Trailing OLS slope of ``gap[column]`` per year over ``window`` months.

    Args:
        gap: Frame with ``ds`` and ``column`` (as returned by :func:`jonas_gap`).
        window: Number of trailing observations in each fit (``>= 3``).
        column: Column to differentiate.

    Returns:
        Slopes indexed by ``ds``; the first ``window − 1`` values are NaN.
    """
    if window < 3:
        raise ValueError("window must be >= 3")
    t = pd.Series(decimal_years(gap["ds"]), index=pd.DatetimeIndex(gap["ds"]))
    y = pd.Series(gap[column].to_numpy(dtype="float64"), index=t.index)
    roll_t = t.rolling(window, min_periods=window)
    cov = (t * y).rolling(window, min_periods=window).mean() - roll_t.mean() * y.rolling(
        window, min_periods=window
    ).mean()
    var = roll_t.var(ddof=0)
    return (cov / var).rename("dgdt")


def _resample_frame(frame: SeriesFrame, rng: np.random.Generator) -> SeriesFrame:
    """One bootstrap draw of the index construction (see module docstring)."""
    comps = side_components(frame)
    parts: list[pd.DataFrame] = []
    z = float(stats.norm.ppf(0.5 + BAND_COVERAGE / 2))
    for ids in comps.values():
        picks = rng.choice(len(ids), size=len(ids), replace=True) if len(ids) > 1 else [0]
        for j, k in enumerate(picks):
            part = frame.loc[frame["series_id"] == ids[k]].copy()
            if "y_lo" in part.columns and "y_hi" in part.columns:
                sigma = (np.log(part["y_hi"]) - np.log(part["y_lo"])) / (2 * z)
                sigma = sigma.fillna(0.0).clip(lower=0.0).to_numpy()
                part["y"] = part["y"] * np.exp(sigma * rng.standard_normal(len(part)))
            part["series_id"] = f"{ids[k]}#{j}"
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


@dataclass(frozen=True)
class _Replicate:
    """Picklable bootstrap task."""

    frame: pd.DataFrame
    base_year: int
    window: int
    seed: int
    weights: Mapping[str, float] | None

    def __call__(self, r: int) -> pd.Series:
        rng = np.random.default_rng(derive_seed(self.seed, "jonas.velocity", r))
        draw = _resample_frame(self.frame, rng)
        weights = None
        if self.weights is not None:
            weights = {
                sid: self.weights.get(sid.split("#")[0], 1.0) for sid in draw["series_id"].unique()
            }
        gap = jonas_gap(draw, self.base_year, weights)
        return gap_slope(gap, self.window)


def leave_one_out(
    frame: SeriesFrame,
    base_year: int,
    window: int = 12,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """``dG/dt`` with each component dropped in turn.

    Returns:
        Wide frame indexed by ``ds`` with one column per dropped ``series_id``
        (empty when every side has a single component).
    """
    cols: dict[str, pd.Series] = {}
    for ids in side_components(frame).values():
        if len(ids) < 2:
            continue
        for sid in ids:
            gap = jonas_gap(frame.loc[frame["series_id"] != sid], base_year, weights)
            cols[sid] = gap_slope(gap, window)
    return pd.DataFrame(cols)


def gap_velocity(
    frame: SeriesFrame,
    base_year: int,
    *,
    window: int = 12,
    n_boot: int = 500,
    level: float = 0.90,
    seed: int | None = None,
    n_jobs: int | None = None,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compute ``dG/dt`` with bootstrap and leave-one-out intervals.

    Args:
        frame: ``SeriesFrame`` with ``"C"`` and ``"H"`` ``source_group`` rows.
        base_year: Normalisation year for :func:`jonas_gap`.
        window: Trailing months per slope fit.
        n_boot: Bootstrap replicates (0 disables the bootstrap).
        level: Two-sided coverage of the bootstrap percentile interval.
        seed: Root seed; defaults to ``Settings.seed``.
        n_jobs: Worker count for :func:`kairos.compute.pmap`.
        weights: Optional component weights.

    Returns:
        Frame with columns :data:`VELOCITY_COLUMNS`, one row per month of ``G``.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    gap = jonas_gap(frame, base_year, weights)
    out = gap.loc[:, ["ds", "gap"]].copy()
    slope = gap_slope(gap, window)
    out["dgdt"] = slope.to_numpy()
    ds_index = pd.DatetimeIndex(out["ds"])

    if n_boot > 0:
        root = get_settings().seed if seed is None else seed
        task = _Replicate(frame, base_year, window, root, weights)
        log.debug("jonas velocity bootstrap: %d replicates", n_boot)
        reps = pmap(task, range(n_boot), n_jobs=n_jobs)
        # A resample may cover fewer months than the full frame; align on ds.
        arr = np.column_stack([s.reindex(ds_index).to_numpy(dtype="float64") for s in reps])
        alpha = (1 - level) / 2
        with warnings.catch_warnings():  # warm-up months are all-NaN by design
            warnings.simplefilter("ignore", RuntimeWarning)
            out["boot_lo"] = np.nanquantile(arr, alpha, axis=1)
            out["boot_hi"] = np.nanquantile(arr, 1 - alpha, axis=1)
            out["boot_se"] = np.nanstd(arr, axis=1, ddof=1)
    else:
        out["boot_lo"] = out["boot_hi"] = out["dgdt"]
        out["boot_se"] = 0.0

    loo = leave_one_out(frame, base_year, window, weights).reindex(ds_index)
    if loo.shape[1]:
        out["loo_min"] = loo.min(axis=1).to_numpy()
        out["loo_max"] = loo.max(axis=1).to_numpy()
    else:
        out["loo_min"] = out["loo_max"] = out["dgdt"]

    out["dgdt_lo"] = out[["dgdt", "boot_lo", "loo_min"]].min(axis=1, skipna=False)
    out["dgdt_hi"] = out[["dgdt", "boot_hi", "loo_max"]].max(axis=1, skipna=False)
    return out.loc[:, list(VELOCITY_COLUMNS)].reset_index(drop=True)
