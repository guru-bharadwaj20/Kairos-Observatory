"""Synthetic capability series with a known amount of self-coupling.

Phase 4 is tested without network access, so tests and demos draw from this
generator. In log space::

    driver_k(t)  = random walk with drift                  (compute, investment, researchers)
    e(t)         = phi * e(t-1) + noise                     (self-coupled residual)
    ln C(t)      = a + b * t + sum_k beta_k * driver_k(t - lag_k) + e(t)

``phi`` (``coupling``) controls how much the target's own past predicts its
future *after* the drivers: ``coupling=0`` gives ``κ ≈ 0``, values near 1 give
a clearly positive ``κ``. Values are returned in raw positive units (``exp``),
following the ``SeriesFrame`` convention.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from kairos.types import SeriesFrame, validate_series_frame

__all__ = ["DEFAULT_DRIVER_LAGS", "TARGET_ID", "synthetic_series"]

TARGET_ID = "capability"
DEFAULT_DRIVER_LAGS: Mapping[str, int] = {"compute": 3, "investment": 6, "researchers": 12}
"""Driver series IDs and the lag (months) at which each drives the target."""


def synthetic_series(
    n: int = 192,
    *,
    coupling: float = 0.8,
    seed: int = 0,
    start: str = "2010-01-01",
    drivers: Mapping[str, int] | None = None,
    noise: float = 0.03,
    driver_beta: float = 0.5,
    events: Mapping[str, tuple[str, float]] | None = None,
    series_id: str = TARGET_ID,
) -> SeriesFrame:
    """Simulate one target series and its exogenous drivers as a long ``SeriesFrame``.

    Args:
        n: Months to simulate.
        coupling: AR(1) coefficient ``phi`` of the residual, in ``[0, 1)``.
        seed: RNG seed.
        start: First month.
        drivers: Driver IDs mapped to their lag; defaults to :data:`DEFAULT_DRIVER_LAGS`.
        noise: Innovation standard deviation (log units).
        driver_beta: Loading of every driver on the target.
        events: Optional ``{name: (date, effect)}`` level shifts (log units) added to
            the target from ``date`` onwards.
        series_id: ID of the target series.

    Returns:
        Long frame with ``ds``, ``y``, ``series_id``, ``source``, ``source_group``, ``unit``.
    """
    if not 0.0 <= coupling < 1.0:
        raise ValueError("coupling must be in [0, 1)")
    drivers = dict(DEFAULT_DRIVER_LAGS if drivers is None else drivers)
    rng = np.random.default_rng(seed)
    max_lag = max(drivers.values(), default=0)
    total = n + max_lag
    ds = pd.date_range(start, periods=n, freq="MS")

    frames: list[pd.DataFrame] = []
    signal = np.zeros(n)
    for name, lag in drivers.items():
        walk = 1.0 + np.cumsum(rng.normal(0.02, 0.04, total))
        signal += driver_beta * walk[max_lag - lag : max_lag - lag + n]
        frames.append(_frame(ds, np.exp(walk[max_lag:]), name, "synthetic-exo"))

    resid = np.zeros(n)
    shocks = rng.normal(0.0, noise, n)
    for t in range(1, n):
        resid[t] = coupling * resid[t - 1] + shocks[t]
    log_target = 2.0 + 0.01 * np.arange(n) + signal + resid
    for when, effect in (events or {}).values():
        log_target = log_target + effect * (ds >= pd.Timestamp(when))
    frames.insert(0, _frame(ds, np.exp(log_target), series_id, "synthetic-c"))
    return validate_series_frame(pd.concat(frames, ignore_index=True))


def _frame(
    ds: pd.DatetimeIndex, values: npt.NDArray[np.float64], series_id: str, group: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": ds,
            "y": values.astype(np.float64),
            "series_id": series_id,
            "source": "synthetic",
            "source_group": group,
            "unit": "index",
        }
    )
