"""K-9.08: scores for forecasts and alarms.

Probabilistic forecasts (score in log units for positive series, D2):

* :func:`crps_gaussian` -- closed-form CRPS of a normal forecast.
* :func:`crps_ensemble` -- CRPS of a sample/ensemble forecast (energy form).
* :func:`pit_values` / :func:`pit_histogram` -- probability integral transform;
  a calibrated forecast gives a flat histogram.
* :func:`interval_coverage` -- share of outcomes inside the forecast interval.

Alarms (from a backtest's :class:`~kairos.validation.detectors.Prediction` list):

* :func:`false_alarm_rate` -- alarms raised when no event followed within the horizon.
* :func:`warning_lead_time` -- time from the first alarm of the final unbroken
  run of alarms to the event.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import numpy.typing as npt
from scipy import stats

from kairos.validation.detectors import Prediction

__all__ = [
    "crps_ensemble",
    "crps_gaussian",
    "false_alarm_rate",
    "interval_coverage",
    "pit_histogram",
    "pit_values",
    "warning_lead_time",
]

ArrayLike = npt.ArrayLike
FloatArray = npt.NDArray[np.float64]


def crps_gaussian(y: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> FloatArray:
    """CRPS of ``N(mu, sigma²)`` at outcomes ``y`` (Gneiting & Raftery, 2007)."""
    y_, mu_, sd = (np.asarray(a, dtype=np.float64) for a in (y, mu, sigma))
    if np.any(sd <= 0):
        raise ValueError("sigma must be positive")
    z = (y_ - mu_) / sd
    crps = sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.asarray(crps, dtype=np.float64)


def crps_ensemble(y: ArrayLike, samples: ArrayLike) -> FloatArray:
    """CRPS of an ensemble: ``E|X - y| - ½·E|X - X'|``.

    Args:
        y: Outcomes, shape ``(n,)``.
        samples: Ensemble members, shape ``(n, m)`` (or ``(m,)`` for one outcome).
    """
    y_ = np.atleast_1d(np.asarray(y, dtype=np.float64))
    x = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    if x.shape[0] != y_.shape[0]:
        raise ValueError("samples must have one row per outcome")
    x = np.sort(x, axis=1)
    m = x.shape[1]
    term1 = np.abs(x - y_[:, None]).mean(axis=1)
    # E|X - X'| from sorted samples in O(m log m).
    weights = 2 * np.arange(1, m + 1) - m - 1
    term2 = (x * weights).sum(axis=1) * 2 / (m * m)
    return np.asarray(term1 - 0.5 * term2, dtype=np.float64)


def pit_values(y: ArrayLike, samples: ArrayLike) -> FloatArray:
    """Probability integral transform of each outcome under its ensemble.

    Ties are split at random-free midpoints: ``(#below + ½·#equal) / m``.
    """
    y_ = np.atleast_1d(np.asarray(y, dtype=np.float64))
    x = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    below = (x < y_[:, None]).sum(axis=1)
    equal = (x == y_[:, None]).sum(axis=1)
    return np.asarray((below + 0.5 * equal) / x.shape[1], dtype=np.float64)


def pit_histogram(pit: ArrayLike, bins: int = 10) -> tuple[FloatArray, float]:
    """Normalised PIT histogram (1.0 = flat) and the chi-square uniformity p-value."""
    values = np.asarray(pit, dtype=np.float64)
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    expected = len(values) / bins
    p_value = float(stats.chisquare(counts).pvalue) if len(values) else float("nan")
    density = counts / expected if expected else counts.astype(np.float64)
    return np.asarray(density, dtype=np.float64), p_value


def interval_coverage(y: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    """Share of outcomes with ``lo <= y <= hi``."""
    y_, lo_, hi_ = (np.asarray(a, dtype=np.float64) for a in (y, lo, hi))
    if y_.size == 0:
        return float("nan")
    return float(((y_ >= lo_) & (y_ <= hi_)).mean())


def _years_between(a: date, b: date) -> float:
    return (b - a).days / 365.25


def false_alarm_rate(
    predictions: Sequence[Prediction], event: date | None, *, horizon_years: float = 1.0
) -> float:
    """Share of alarms not followed by ``event`` within ``horizon_years``.

    With ``event=None`` (no event ever happened, e.g. Moore's law) every alarm is
    false. Returns ``nan`` when no alarm was raised.
    """
    alarms = [p for p in predictions if p.alarm]
    if not alarms:
        return float("nan")
    if event is None:
        return 1.0
    false = [p for p in alarms if not 0 <= _years_between(p.as_of, event) <= horizon_years]
    return len(false) / len(alarms)


def warning_lead_time(predictions: Sequence[Prediction], event: date) -> float | None:
    """Years from the start of the last unbroken pre-event alarm run to ``event``.

    Returns ``None`` if the last prediction before the event did not alarm.
    """
    before = sorted((p for p in predictions if p.as_of <= event), key=lambda p: p.as_of)
    if not before or not before[-1].alarm:
        return None
    start = before[-1].as_of
    for p in reversed(before):
        if not p.alarm:
            break
        start = p.as_of
    return _years_between(start, event)
