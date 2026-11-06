"""Detector contract for historical validation, plus a self-contained baseline.

The validation harness never imports a concrete detector directly. It talks to
anything that satisfies :class:`Detector`: a callable that takes the data visible
at one as-of date and returns a :class:`Prediction` (a critical time ``t_c`` with
an interval, the exponent ``δ`` and an alarm level).

The production detectors live in :mod:`kairos.accel` (Phase 5). Until they land,
:class:`HyperbolicDetector` is a small, honest baseline: a profile least-squares
fit of ``ln x = a - k·ln(t_c - t)`` (so ``δ = 1/k``, D5) compared by AIC with a
plain exponential. Plugging in Phase 5 is a one-line change to
:data:`DETECTORS`.

Time is handled as **decimal years** (``2000.25`` is 1 April 2000), which keeps
``t_c`` comparable across monthly, weekly and annual data.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from kairos.types import DialReading, SeriesFrame

__all__ = [
    "DETECTORS",
    "Detector",
    "HyperbolicDetector",
    "Prediction",
    "decimal_year",
    "from_decimal_year",
    "get_detector",
    "single_series_id",
]

FloatArray = npt.NDArray[np.float64]


def decimal_year(ds: pd.Series | pd.DatetimeIndex | date | pd.Timestamp) -> Any:
    """Convert dates to decimal years (day-of-year resolution).

    Accepts a scalar date/timestamp (returns ``float``) or a Series/Index of
    timestamps (returns a float64 NumPy array).
    """
    if isinstance(ds, date | pd.Timestamp):
        ts = pd.Timestamp(ds)
        days = 366.0 if ts.is_leap_year else 365.0
        return float(ts.year + (ts.dayofyear - 1) / days)
    idx = pd.DatetimeIndex(ds)
    ndays = np.where(idx.is_leap_year, 366.0, 365.0)
    return np.asarray(idx.year + (idx.dayofyear - 1) / ndays, dtype=np.float64)


def from_decimal_year(t: float) -> date:
    """Inverse of :func:`decimal_year` for a scalar, rounded to the day."""
    year = math.floor(t)
    days = 366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365
    stamp = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=round((t - year) * days))
    return date(stamp.year, stamp.month, stamp.day)


@dataclass(frozen=True, slots=True)
class Prediction:
    """What a detector says about one series at one as-of date.

    Attributes:
        detector: Detector name.
        series_id: Series the prediction is about.
        as_of: Data vintage: the detector saw nothing after this date.
        n_obs: Observations the detector used.
        t_c: Point estimate of the critical time in decimal years (``None`` if no
            finite-time singularity was fitted).
        t_c_lo: Lower end of the ``t_c`` interval.
        t_c_hi: Upper end of the ``t_c`` interval (``inf`` if unbounded).
        delta: Hyperbolic exponent ``δ`` (``None`` when not applicable).
        alarm: Whether the detector raises an alarm at this date.
        level: Tripwire-style level 0 (Normal) to 4.
        detail: Diagnostics (fit statistics, filters, ...).
    """

    detector: str
    series_id: str
    as_of: date
    n_obs: int
    t_c: float | None
    t_c_lo: float | None
    t_c_hi: float | None
    delta: float | None
    alarm: bool
    level: int
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def lead_time_years(self) -> float | None:
        """``t_c - as_of`` in years, or ``None`` without a ``t_c``."""
        if self.t_c is None:
            return None
        return float(self.t_c - decimal_year(self.as_of))

    def to_dial_reading(self) -> DialReading:
        """Express the prediction as an acceleration-dial reading (value = ``t_c``)."""
        nan = float("nan")
        value = self.t_c if self.t_c is not None else nan
        lo = self.t_c_lo if self.t_c_lo is not None else value
        hi = self.t_c_hi if self.t_c_hi is not None else value
        return DialReading(
            dial="acceleration",
            value=value,
            level=self.level,
            ci_lo=lo,
            ci_hi=hi,
            as_of=self.as_of,
            detail={"detector": self.detector, "delta": self.delta, **self.detail},
        )

    def as_record(self) -> dict[str, Any]:
        """Flat dictionary for building a results table."""
        record = asdict(self)
        record.pop("detail")
        record["lead_time_years"] = self.lead_time_years
        return record


@runtime_checkable
class Detector(Protocol):
    """Anything that turns an as-of view of one series into a :class:`Prediction`.

    Implementations must be picklable (module-level classes or functions) so the
    backtest runner can fan them out over the CPU pool, and must use only the
    rows they are given: the runner has already removed every row that was not
    available at ``as_of``.
    """

    @property
    def name(self) -> str:
        """Detector name used in results."""
        ...

    def __call__(self, frame: SeriesFrame, as_of: date) -> Prediction:
        """Return the prediction for ``frame`` (one series) as of ``as_of``."""
        ...


def single_series_id(frame: SeriesFrame) -> str:
    """Return the only ``series_id`` in ``frame`` or raise ``ValueError``."""
    ids = frame["series_id"].unique()
    if len(ids) != 1:
        raise ValueError(f"detector expects one series, got {list(ids)}")
    return str(ids[0])


def _aic(sse: float, n: int, p: int) -> float:
    return n * math.log(max(sse, 1e-300) / n) + 2 * p


def _ols_sse(x: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    """Return ``(intercept, slope, sse)`` for ``y ~ x``."""
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    slope = float(((x - xm) * (y - ym)).sum() / sxx) if sxx > 0 else 0.0
    intercept = float(ym - slope * xm)
    sse = float(((y - intercept - slope * x) ** 2).sum())
    return intercept, slope, sse


@dataclass(frozen=True)
class HyperbolicDetector:
    """Baseline finite-time-singularity detector (hyperbolic vs exponential).

    Fits ``ln x = a - k·ln(t_c - t)`` by profiling the sum of squared errors over
    a dense grid of ``t_c`` (each cell is an ordinary least-squares solve), then
    refines around the best cell. The exponent is ``δ = 1/k``. The ``t_c``
    interval is the profile-likelihood set where the F-statistic against the
    best fit stays below its ``level`` quantile. Errors are treated as
    independent, so the interval is **too narrow** for autocorrelated data
    (D4); the production detector bootstraps instead.

    An alarm needs all three: a hyperbola that beats the exponential by more than
    ``aic_margin``, ``k > 0``, and ``t_c`` no more than ``horizon_years`` past
    the last observation.

    Attributes:
        name: Detector name used in results.
        horizon_years: Farthest ``t_c`` (from the last point) that can alarm.
        aic_margin: AIC advantage the hyperbola needs over the exponential.
        min_points: Fewer observations than this give a no-alarm prediction.
        grid_size: ``t_c`` grid cells.
        max_span_factor: The grid reaches ``t_last + max_span_factor·span``.
        level: Confidence level of the ``t_c`` interval.
    """

    name: str = "hyperbolic"
    horizon_years: float = 50.0
    aic_margin: float = 10.0
    min_points: int = 8
    grid_size: int = 600
    max_span_factor: float = 20.0
    level: float = 0.95

    def fit(self, t: FloatArray, y: FloatArray) -> dict[str, float]:
        """Fit both models to decimal years ``t`` and raw values ``y``."""
        ly = np.log(y)
        t_last = float(t.max())
        span = max(float(t.max() - t.min()), 1e-6)
        gaps = np.geomspace(span * 1e-3, span * self.max_span_factor, self.grid_size)
        sse = np.array([_ols_sse(np.log(t_last + g - t), ly)[2] for g in gaps])
        best = int(np.argmin(sse))
        lo_g = gaps[max(best - 1, 0)]
        hi_g = gaps[min(best + 1, len(gaps) - 1)]
        fine = np.linspace(lo_g, hi_g, 101)
        fine_sse = np.array([_ols_sse(np.log(t_last + g - t), ly)[2] for g in fine])
        gap = float(fine[int(np.argmin(fine_sse))])
        a, slope, sse_h = _ols_sse(np.log(t_last + gap - t), ly)
        k = -slope
        _, growth, sse_e = _ols_sse(t, ly)
        n = len(t)
        dof = max(n - 3, 1)
        threshold = sse_h * (1.0 + float(stats.f.ppf(self.level, 1, dof)) / dof)
        all_gaps = np.concatenate([gaps, fine])
        all_sse = np.concatenate([sse, fine_sse])
        inside = all_gaps[all_sse <= threshold]
        upper_open = bool(sse[-1] <= threshold)
        return {
            "t_c": t_last + gap,
            "t_c_lo": t_last + float(inside.min()) if inside.size else t_last + gap,
            "t_c_hi": math.inf if upper_open else t_last + float(inside.max()),
            "k": k,
            "a": a,
            "sse_hyperbolic": sse_h,
            "sse_exponential": sse_e,
            "growth_rate": growth,
            "aic_hyperbolic": _aic(sse_h, n, 3),
            "aic_exponential": _aic(sse_e, n, 2),
        }

    def __call__(self, frame: SeriesFrame, as_of: date) -> Prediction:
        """Fit the series in ``frame`` and turn the fit into a prediction."""
        series_id = single_series_id(frame)
        data = frame.dropna(subset=["y"]).sort_values("ds")
        n = len(data)
        if n < self.min_points:
            return Prediction(
                self.name, series_id, as_of, n, None, None, None, None, False, 0,
                {"reason": f"fewer than {self.min_points} points"},
            )  # fmt: skip
        t = decimal_year(data["ds"])
        y = data["y"].to_numpy(dtype=np.float64)
        fit = self.fit(t, y)
        k = fit["k"]
        delta = 1.0 / k if k > 0 else None
        aic_gain = fit["aic_exponential"] - fit["aic_hyperbolic"]
        lead = fit["t_c"] - float(t.max())
        alarm = bool(k > 0 and aic_gain > self.aic_margin and lead <= self.horizon_years)
        detail = {"aic_gain": aic_gain, **{key: fit[key] for key in ("k", "growth_rate")}}
        return Prediction(
            detector=self.name,
            series_id=series_id,
            as_of=as_of,
            n_obs=n,
            t_c=fit["t_c"] if k > 0 else None,
            t_c_lo=fit["t_c_lo"] if k > 0 else None,
            t_c_hi=fit["t_c_hi"] if k > 0 else None,
            delta=delta,
            alarm=alarm,
            level=_level(alarm, lead),
            detail=detail,
        )


def _level(alarm: bool, lead_years: float) -> int:
    """Provisional level map until the Phase 8 tripwire ladder is available."""
    if not alarm:
        return 0
    if lead_years <= 2:
        return 3
    if lead_years <= 10:
        return 2
    return 1


DETECTORS: dict[str, Callable[..., Detector]] = {
    "hyperbolic": HyperbolicDetector,
}
"""Registry of detector factories. Phase 5 adds/overrides entries here."""


def get_detector(name: str, **kwargs: Any) -> Detector:
    """Instantiate a registered detector by name."""
    try:
        factory = DETECTORS[name]
    except KeyError:
        raise KeyError(f"unknown detector {name!r}; known: {sorted(DETECTORS)}") from None
    return factory(**kwargs)
