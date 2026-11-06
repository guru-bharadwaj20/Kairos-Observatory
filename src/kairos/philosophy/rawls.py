"""Rawls's veil of ignorance as a maximin rule over the scenario ensemble.

Behind the veil we don't know which world we are in, so we choose the act that does
best in the *worst plausible* world, and report the expected-value choice next to it
so that a reader sees when the two disagree.

Two entry points:

* :func:`maximin` -- the classic rule over a payoff table ``action -> utilities per
  scenario``.
* :func:`worst_case_p_danger` -- the governor's use: ``P(danger)`` in each ensemble
  scenario (``DialReading.detail["scenarios"]``: ``name -> draws``), the worst plausible
  one, and the weight-averaged one.

A scenario is *plausible* when its normalised weight is at least ``min_weight``
(``policy.yaml``, section ``rawls``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from kairos.philosophy.probability import DangerRule, p_danger_from_samples
from kairos.types import DialReading

__all__ = [
    "SCENARIOS_KEY",
    "WEIGHTS_KEY",
    "MaximinDecision",
    "ScenarioPDanger",
    "maximin",
    "plausible",
    "worst_case_p_danger",
]

SCENARIOS_KEY = "scenarios"
WEIGHTS_KEY = "scenario_weights"


def _normalised(names: Sequence[str], weights: Mapping[str, float] | None) -> dict[str, float]:
    if not names:
        raise ValueError("need at least one scenario")
    raw = {n: 1.0 if weights is None else float(weights.get(n, 0.0)) for n in names}
    if any(w < 0 for w in raw.values()):
        raise ValueError("scenario weights must be >= 0")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("scenario weights sum to zero")
    return {n: w / total for n, w in raw.items()}


def plausible(weights: Mapping[str, float], min_weight: float) -> list[str]:
    """Names whose normalised weight is >= ``min_weight``; the heaviest if none is."""
    keep = [n for n, w in weights.items() if w >= min_weight]
    return keep or [max(weights, key=lambda n: weights[n])]


class MaximinDecision(BaseModel):
    """Maximin choice reported next to the expected-value choice.

    Attributes:
        maximin_action: Act with the best worst-plausible utility.
        maximin_value: That worst-plausible utility.
        worst_scenario: Scenario in which the maximin act attains its worst case.
        ev_action: Act with the best weighted mean utility.
        ev_value: That mean utility.
        agree: Whether the two rules pick the same act.
        plausible: Scenarios considered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    maximin_action: str
    maximin_value: float
    worst_scenario: str
    ev_action: str
    ev_value: float
    agree: bool
    plausible: list[str]


def maximin(
    payoffs: Mapping[str, Mapping[str, float]],
    *,
    weights: Mapping[str, float] | None = None,
    min_weight: float = 0.0,
) -> MaximinDecision:
    """Choose an act by maximin over plausible scenarios (ties: first act listed).

    Args:
        payoffs: ``action -> {scenario: utility}``; every act must cover every scenario.
        weights: Scenario weights (default equal).
        min_weight: Plausibility floor on normalised weights.
    """
    if not payoffs:
        raise ValueError("need at least one action")
    scenarios = list(next(iter(payoffs.values())))
    for act, row in payoffs.items():
        if set(row) != set(scenarios):
            raise ValueError(f"action {act!r} does not cover the same scenarios")
    w = _normalised(scenarios, weights)
    keep = plausible(w, min_weight)

    worst = {a: min(keep, key=payoffs[a].__getitem__) for a in payoffs}
    mm = max(payoffs, key=lambda a: payoffs[a][worst[a]])
    ev = {a: sum(w[s] * payoffs[a][s] for s in scenarios) for a in payoffs}
    ev_best = max(payoffs, key=lambda a: ev[a])
    return MaximinDecision(
        maximin_action=mm,
        maximin_value=payoffs[mm][worst[mm]],
        worst_scenario=worst[mm],
        ev_action=ev_best,
        ev_value=ev[ev_best],
        agree=mm == ev_best,
        plausible=keep,
    )


class ScenarioPDanger(BaseModel):
    """``P(danger)`` across the scenario ensemble of one dial.

    Attributes:
        dial: Dial name.
        per_scenario: ``P(danger)`` in each scenario.
        weights: Normalised scenario weights.
        worst: ``P(danger)`` in the worst plausible scenario (the Rawls figure).
        worst_scenario: Name of that scenario.
        expected: Weight-averaged ``P(danger)`` (the expected-value figure).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dial: str
    per_scenario: dict[str, float]
    weights: dict[str, float]
    worst: float = Field(ge=0, le=1)
    worst_scenario: str
    expected: float = Field(ge=0, le=1)


def worst_case_p_danger(
    reading: DialReading, rule: DangerRule, *, min_weight: float = 0.0
) -> ScenarioPDanger:
    """Per-scenario, worst-plausible and expected ``P(danger)`` for one reading.

    Raises:
        ValueError: If the reading has no ``detail["scenarios"]``.
    """
    scenarios: Mapping[str, Sequence[float]] | None = reading.detail.get(SCENARIOS_KEY)
    if not scenarios:
        raise ValueError(f"dial {reading.dial!r} has no detail[{SCENARIOS_KEY!r}]")
    w = _normalised(list(scenarios), reading.detail.get(WEIGHTS_KEY))
    per = {s: p_danger_from_samples(reading.dial, x, rule).p for s, x in scenarios.items()}
    keep = plausible(w, min_weight)
    worst = max(keep, key=lambda s: per[s])
    expected = float(np.clip(sum(w[s] * per[s] for s in per), 0.0, 1.0))
    return ScenarioPDanger(
        dial=reading.dial,
        per_scenario=per,
        weights=w,
        worst=per[worst],
        worst_scenario=worst,
        expected=expected,
    )
