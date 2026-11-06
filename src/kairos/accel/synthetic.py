"""Synthetic test series with known growth law, ``δ`` and ``t_c`` (K-5.01).

Every generator returns a :class:`SyntheticSeries`: a monthly ``SeriesFrame`` in raw
positive units plus the ground truth used to build it. Time is measured in decimal
years (:func:`decimal_years`), the unit of every ``t`` and ``t_c`` in
:mod:`kairos.accel`.

Growth laws, written for ``ln x(t)``:

* exponential (``δ = 0``): ``ln x = ln x0 + r (t − t0)``
* hyperbolic (finite-time singularity): ``ln x = ln x0 − (1/δ) ln((t_c − t)/(t_c − t0))``,
  so ``g = d ln x/dt = 1/(δ (t_c − t))`` and ``ln g = const + δ ln x``
* logistic: ``x = floor + (cap − floor) / (1 + exp(−r (t − t_mid)))`` (``δ < 0``)

Noise is multiplicative: ``ln y = ln x + ε`` with ``ε`` a stationary AR(1) process of
marginal standard deviation ``noise`` and lag-one autocorrelation ``ar``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from kairos.types import SeriesFrame

__all__ = [
    "FloatArray",
    "SyntheticSeries",
    "ar1_noise",
    "decimal_years",
    "exponential_series",
    "hyperbolic_series",
    "logistic_series",
    "month_starts",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticSeries:
    """A synthetic series and its ground truth.

    Attributes:
        frame: Monthly ``SeriesFrame`` (raw units, noise applied).
        t: Decimal-year time of each row.
        log_x_true: Noise-free ``ln x(t)``.
        law: ``"exponential"``, ``"hyperbolic"``, ``"logistic"`` or ``"lppls"``.
        params: Generator parameters (``delta``, ``t_c``, ...).
    """

    frame: SeriesFrame
    t: FloatArray
    log_x_true: FloatArray
    law: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def log_y(self) -> FloatArray:
        """Observed ``ln y``."""
        y: FloatArray = self.frame["y"].to_numpy(dtype=np.float64)
        return np.log(y)


def month_starts(start: str | pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """``n`` consecutive month-start dates beginning at the month of ``start``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    first = pd.Timestamp(start).to_period("M").to_timestamp()
    return pd.date_range(first, periods=n, freq="MS")


def decimal_years(ds: pd.Series | pd.DatetimeIndex | npt.ArrayLike) -> FloatArray:
    """Convert dates to decimal years (elapsed fraction of the calendar year)."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(ds)))
    years = np.asarray(idx.year, dtype=np.int64)
    start = pd.to_datetime(pd.Series(years).astype(str), format="%Y").to_numpy()
    stop = pd.to_datetime(pd.Series(years + 1).astype(str), format="%Y").to_numpy()
    frac = (idx.to_numpy() - start) / (stop - start)
    return years.astype(np.float64) + np.asarray(frac, dtype=np.float64)


def ar1_noise(n: int, sd: float, ar: float, rng: np.random.Generator) -> FloatArray:
    """Stationary AR(1) noise with marginal standard deviation ``sd``.

    Args:
        n: Length.
        sd: Marginal standard deviation (0 gives zeros).
        ar: Lag-one autocorrelation in ``(-1, 1)``.
        rng: Random generator.
    """
    if not -1.0 < ar < 1.0:
        raise ValueError("ar must lie in (-1, 1)")
    if sd < 0:
        raise ValueError("sd must be >= 0")
    out = np.empty(n, dtype=np.float64)
    innov = rng.standard_normal(n) * sd * np.sqrt(1.0 - ar * ar)
    out[0] = rng.standard_normal() * sd
    for i in range(1, n):
        out[i] = ar * out[i - 1] + innov[i]
    return out


def _finish(
    law: str,
    ds: pd.DatetimeIndex,
    t: FloatArray,
    log_x: FloatArray,
    *,
    noise: float,
    ar: float,
    seed: int,
    series_id: str | None,
    source_group: str,
    params: dict[str, Any],
) -> SyntheticSeries:
    rng = np.random.default_rng(seed)
    log_y = log_x + ar1_noise(len(t), noise, ar, rng)
    frame = pd.DataFrame(
        {
            "ds": ds,
            "y": np.exp(log_y),
            "series_id": series_id or f"synthetic_{law}",
            "source": "synthetic",
            "source_group": source_group,
            "unit": "arb",
        }
    )
    params = {**params, "noise": noise, "ar": ar, "seed": seed}
    return SyntheticSeries(frame=frame, t=t, log_x_true=log_x, law=law, params=params)


def exponential_series(
    n: int = 240,
    *,
    start: str = "2000-01-01",
    x0: float = 1.0,
    rate: float = 0.35,
    noise: float = 0.05,
    ar: float = 0.0,
    seed: int = 0,
    series_id: str | None = None,
    source_group: str = "synthetic",
) -> SyntheticSeries:
    """Exponential growth ``x = x0·exp(rate·(t − t0))``: constant ``g``, so ``δ = 0``.

    Args:
        n: Number of monthly points.
        start: First month.
        x0: Value at the first month.
        rate: Growth rate per year.
        noise: Marginal sd of the log noise.
        ar: AR(1) coefficient of the log noise.
        seed: Noise seed.
        series_id: Defaults to ``synthetic_exponential``.
        source_group: Provenance group written to the frame.
    """
    ds = month_starts(start, n)
    t = decimal_years(ds)
    log_x = np.log(x0) + rate * (t - t[0])
    return _finish(
        "exponential",
        ds,
        t,
        log_x,
        noise=noise,
        ar=ar,
        seed=seed,
        series_id=series_id,
        source_group=source_group,
        params={"x0": x0, "rate": rate, "delta": 0.0, "t_c": float("inf")},
    )


def hyperbolic_series(
    n: int = 240,
    *,
    start: str = "2000-01-01",
    x0: float = 1.0,
    delta: float = 0.5,
    t_c: float = 2025.0,
    noise: float = 0.05,
    ar: float = 0.0,
    seed: int = 0,
    series_id: str | None = None,
    source_group: str = "synthetic",
) -> SyntheticSeries:
    """Hyperbolic growth ``x = x0·((t_c − t)/(t_c − t0))^(−1/δ)``, singular at ``t_c``.

    Args:
        n: Number of monthly points; the last must lie before ``t_c``.
        start: First month.
        x0: Value at the first month.
        delta: Exponent ``δ > 0`` in ``g ∝ x^δ``.
        t_c: Critical time in decimal years.
        noise: Marginal sd of the log noise.
        ar: AR(1) coefficient of the log noise.
        seed: Noise seed.
        series_id: Defaults to ``synthetic_hyperbolic``.
        source_group: Provenance group written to the frame.

    Raises:
        ValueError: If ``delta <= 0`` or the series reaches ``t_c``.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0 for a hyperbolic series")
    ds = month_starts(start, n)
    t = decimal_years(ds)
    if t[-1] >= t_c:
        raise ValueError(f"last point {t[-1]:.3f} must lie before t_c={t_c}")
    log_x = np.log(x0) - np.log((t_c - t) / (t_c - t[0])) / delta
    return _finish(
        "hyperbolic",
        ds,
        t,
        log_x,
        noise=noise,
        ar=ar,
        seed=seed,
        series_id=series_id,
        source_group=source_group,
        params={"x0": x0, "delta": delta, "t_c": t_c},
    )


def logistic_series(
    n: int = 240,
    *,
    start: str = "2000-01-01",
    floor: float = 1.0,
    cap: float = 1000.0,
    rate: float = 0.6,
    t_mid: float = 2010.0,
    noise: float = 0.05,
    ar: float = 0.0,
    seed: int = 0,
    series_id: str | None = None,
    source_group: str = "synthetic",
) -> SyntheticSeries:
    """Logistic growth from ``floor`` to ``cap``: ``g`` falls as ``x`` grows, so ``δ < 0``.

    Args:
        n: Number of monthly points.
        start: First month.
        floor: Lower asymptote (> 0, keeps ``x`` positive).
        cap: Upper asymptote.
        rate: Logistic rate per year.
        t_mid: Inflection time in decimal years.
        noise: Marginal sd of the log noise.
        ar: AR(1) coefficient of the log noise.
        seed: Noise seed.
        series_id: Defaults to ``synthetic_logistic``.
        source_group: Provenance group written to the frame.
    """
    if not 0 < floor < cap:
        raise ValueError("need 0 < floor < cap")
    ds = month_starts(start, n)
    t = decimal_years(ds)
    x = floor + (cap - floor) / (1.0 + np.exp(-rate * (t - t_mid)))
    return _finish(
        "logistic",
        ds,
        t,
        np.log(x),
        noise=noise,
        ar=ar,
        seed=seed,
        series_id=series_id,
        source_group=source_group,
        params={"floor": floor, "cap": cap, "rate": rate, "t_mid": t_mid, "t_c": float("inf")},
    )
