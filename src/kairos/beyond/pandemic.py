"""Pandemic growth against health-system capacity.

The epidemic analogue of the Jonas Gap: ``C(t)`` is hospital demand and
``H(t)`` is the number of beds. ``G = ln C − ln H``; ``G = 0`` is the date the
system is overwhelmed. Lead time is how long until the upper growth scenario
reaches ``G = 0``, which is the Collingridge window for a lockdown decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["CapacityGap", "capacity_gap", "simulate_sir"]

FloatArray = NDArray[np.float64]


def simulate_sir(
    days: int,
    *,
    population: float = 1e7,
    beta: float = 0.3,
    gamma: float = 0.1,
    initial_infected: float = 10.0,
    lockdown_day: int | None = None,
    lockdown_factor: float = 0.4,
) -> FloatArray:
    """Daily new infections from a discrete-time SIR model.

    ``lockdown_day`` multiplies ``beta`` by ``lockdown_factor`` from that day on,
    which is how an intervention event bends the curve.
    """
    s, i = population - initial_infected, initial_infected
    out = np.empty(days)
    for d in range(days):
        b = beta * (lockdown_factor if lockdown_day is not None and d >= lockdown_day else 1.0)
        new = b * s * i / population
        s, i = s - new, i + new - gamma * i
        out[d] = new
    return out


@dataclass(frozen=True)
class CapacityGap:
    """Gap between hospital demand and capacity.

    Attributes:
        gap: ``ln(demand) − ln(capacity)`` per day.
        velocity: ``dG/dt`` per day, from a trailing linear fit.
        growth_low: Lower bound of the recent log growth rate per day.
        growth_high: Upper bound of the recent log growth rate per day.
        lead_time_days: Days until ``G = 0`` under ``growth_high``; 0 if already
            over capacity, ``inf`` if the gap is not closing.
    """

    gap: FloatArray
    velocity: float
    growth_low: float
    growth_high: float
    lead_time_days: float


def capacity_gap(
    infections: ArrayLike,
    capacity: ArrayLike | float,
    *,
    hospital_rate: float = 0.05,
    length_of_stay: int = 10,
    window: int = 14,
    z: float = 1.645,
) -> CapacityGap:
    """Compare hospital occupancy implied by ``infections`` with ``capacity``.

    Occupancy is ``hospital_rate`` times the trailing ``length_of_stay``-day sum
    of infections. The growth interval is ``z`` standard errors around the slope
    of ``ln occupancy`` over the last ``window`` days (``z = 1.645`` gives 90%).
    """
    inf = np.asarray(infections, dtype=np.float64)
    if inf.size < window + length_of_stay:
        raise ValueError("series too short for the chosen window and length of stay")
    cap = np.broadcast_to(np.asarray(capacity, dtype=np.float64), inf.shape)
    occupancy = hospital_rate * np.convolve(inf, np.ones(length_of_stay))[: inf.size]
    gap = np.log(np.maximum(occupancy, 1e-12)) - np.log(cap)

    t = np.arange(window, dtype=np.float64)
    recent = gap[-window:]
    slope, intercept = np.polyfit(t, recent, 1)
    resid = recent - (slope * t + intercept)
    se = float(np.sqrt(resid @ resid / (window - 2) / ((t - t.mean()) @ (t - t.mean()))))
    lo, hi = float(slope - z * se), float(slope + z * se)

    g_now = float(gap[-1])
    if g_now >= 0:
        lead = 0.0
    elif hi <= 0:
        lead = float("inf")
    else:
        lead = -g_now / hi
    return CapacityGap(gap, float(slope), lo, hi, lead)
