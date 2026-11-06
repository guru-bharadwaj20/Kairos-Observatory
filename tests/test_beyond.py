"""Tests for the beyond-AI example: pandemic capacity gap and CSD dial."""

from __future__ import annotations

import math
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from kairos.beyond.csd import CSDConfig, csd_dial, kendall_tau, rolling_indicators  # noqa: E402
from kairos.beyond.pandemic import capacity_gap, simulate_sir  # noqa: E402


def _ar1_series(phis: Any, seed: int = 0) -> Any:
    rng = np.random.default_rng(seed)
    phi = np.asarray(phis)
    x = np.zeros(phi.size)
    for i in range(1, phi.size):
        x[i] = phi[i] * x[i - 1] + rng.normal()
    return x


def test_kendall_tau_extremes() -> None:
    assert kendall_tau([1, 2, 3, 4]) == 1.0
    assert kendall_tau([4, 3, 2, 1]) == -1.0
    assert math.isnan(kendall_tau([1.0]))


def test_rolling_indicators_window_validation() -> None:
    with pytest.raises(ValueError, match="window"):
        rolling_indicators([1.0, 2.0], 3)


def test_csd_detects_approach_to_tipping_point() -> None:
    approaching = _ar1_series(np.linspace(0.1, 0.97, 600))
    dial = csd_dial(approaching, CSDConfig(tau_threshold=0.3))
    assert dial.tau_ar1 > 0.3
    assert dial.tau_variance > 0.3
    assert dial.state == "danger"


def test_csd_stays_calm_on_stationary_noise() -> None:
    stationary = _ar1_series(np.full(600, 0.3), seed=1)
    assert csd_dial(stationary, CSDConfig(tau_threshold=0.5)).state == "calm"


def test_lockdown_bends_the_curve() -> None:
    free = simulate_sir(120)
    locked = simulate_sir(120, lockdown_day=40)
    assert locked.sum() < 0.5 * free.sum()


def test_capacity_gap_lead_time() -> None:
    early = simulate_sir(45)
    gap = capacity_gap(early, capacity=50_000.0)
    assert gap.gap[-1] < 0
    assert gap.velocity > 0
    assert gap.growth_low <= gap.velocity <= gap.growth_high
    assert 0 < gap.lead_time_days < 120
    over = capacity_gap(simulate_sir(90), capacity=10.0)
    assert over.lead_time_days == 0.0


def test_capacity_gap_needs_enough_data() -> None:
    with pytest.raises(ValueError, match="too short"):
        capacity_gap(np.ones(5), capacity=1.0)
