"""Tripwire aggregator: four dials in, one level 0-4 out.

Per dial, every week:

1. ``P(danger)`` from the reading's draws (K-8.03). If the reading carries a scenario
   ensemble, the decision uses the **worst plausible** scenario (Rawls, K-8.04) and
   the weighted mean is reported beside it.
2. The decision probability is fed to the dial's hysteresis state machine (K-8.01),
   whose upper edge is the Jonas threshold ``1/(1+λ)`` (K-8.02).
3. A dial in ``DANGER`` is *confirmed* when its reading reports at least
   ``min_independent_groups`` independent source groups (D10).

The level follows ``tripwires.yaml`` (:func:`level_for`)::

    0 Normal   no dial in danger
    1 Watch    one dial
    2 Warning  two dials, or one dial confirmed
    3 Alarm    three dials
    4 Kairos   all four dials
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kairos.logging import get_logger
from kairos.philosophy.jonas import PolicyChange, PolicySpec, load_policy, read_yaml
from kairos.philosophy.probability import (
    SAMPLES_KEY,
    DangerRule,
    Method,
    danger_rules,
    p_danger,
)
from kairos.philosophy.rawls import SCENARIOS_KEY, worst_case_p_danger
from kairos.tripwires.state import DialState, DialStateMachine
from kairos.types import DialReading

__all__ = [
    "DEFAULT_SPEC",
    "MAX_LEVEL",
    "Confirmation",
    "DialAssessment",
    "Governor",
    "LevelSpec",
    "TripwireAlert",
    "TripwireSpec",
    "level_for",
    "load_tripwire_spec",
]

log = get_logger(__name__)

DEFAULT_SPEC = "tripwires.yaml"
MAX_LEVEL = 4


def level_for(n_danger: int, n_confirmed: int = 0) -> int:
    """Tripwire level from the number of dials in danger and of those confirmed.

    Raises:
        ValueError: If the counts are negative, exceed four or ``n_confirmed > n_danger``.
    """
    if not 0 <= n_confirmed <= n_danger <= MAX_LEVEL:
        raise ValueError(f"need 0 <= n_confirmed <= n_danger <= 4, got {n_confirmed}, {n_danger}")
    if n_danger == 1 and n_confirmed >= 1:
        return 2
    return n_danger


class LevelSpec(BaseModel):
    """One tripwire level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: int = Field(ge=0, le=MAX_LEVEL)
    name: str
    trigger: str
    response: str


class Confirmation(BaseModel):
    """When a dial in danger counts as confirmed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detail_key: str = "n_independent_groups"
    min_independent_groups: int = Field(default=3, ge=2)


class TripwireSpec(BaseModel):
    """Contents of ``tripwires.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    changelog: list[PolicyChange] = Field(default_factory=list)
    dials: list[str]
    confirmation: Confirmation = Field(default_factory=Confirmation)
    levels: list[LevelSpec]

    @model_validator(mode="after")
    def _complete(self) -> TripwireSpec:
        if len(self.dials) != MAX_LEVEL or len(set(self.dials)) != MAX_LEVEL:
            raise ValueError("tripwires.yaml must list exactly four distinct dials")
        if sorted(lv.level for lv in self.levels) != list(range(MAX_LEVEL + 1)):
            raise ValueError("tripwires.yaml must define each level 0-4 exactly once")
        return self

    def level(self, n: int) -> LevelSpec:
        """The spec entry for level ``n``."""
        return next(lv for lv in self.levels if lv.level == n)


def load_tripwire_spec(path: Path | None = None) -> TripwireSpec:
    """Load ``tripwires.yaml`` (default: the packaged copy)."""
    return TripwireSpec.model_validate(read_yaml(path, "kairos.tripwires", DEFAULT_SPEC))


class DialAssessment(BaseModel):
    """The governor's view of one dial this week.

    Attributes:
        dial: Dial name.
        p_danger: Probability used for the decision (worst plausible scenario if any).
        p_expected: Expected-value ``P(danger)`` (pooled draws or weighted scenarios).
        method: How ``P(danger)`` was estimated.
        worst_scenario: Scenario behind ``p_danger``, if an ensemble was given.
        alarm: Whether ``p_danger`` alone exceeds the Jonas threshold this week.
        state: Hysteresis state after this reading.
        confirmed: In danger and seen in enough independent source groups.
        n_independent_groups: Groups reported by the dial, if any.
        as_of: Data vintage of the reading.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dial: str
    p_danger: float = Field(ge=0, le=1)
    p_expected: float = Field(ge=0, le=1)
    method: Method | None
    worst_scenario: str | None = None
    alarm: bool
    state: DialState
    confirmed: bool
    n_independent_groups: int | None = None
    as_of: date | None = None

    @property
    def in_danger(self) -> bool:
        """Whether the dial is in the danger state."""
        return self.state is DialState.DANGER


class TripwireAlert(BaseModel):
    """Output of the governor: the tripwire level and why.

    Attributes:
        level: 0 (Normal) to 4 (Kairos).
        name: Level name from the spec.
        trigger: Trigger text from the spec.
        response: Recommended response from the spec (humans decide).
        as_of: Latest data vintage among the readings.
        n_danger: Dials in danger.
        n_confirmed: Dials in danger and confirmed.
        dials: Per-dial assessments, in spec order.
        missing: Spec dials with no reading this week (held in their last state).
        ignored: Readings of dials outside the spec (reported, not counted).
        spec_version: ``tripwires.yaml`` version.
        policy_version: ``policy.yaml`` version.
        lam: Jonas ``λ``.
        threshold: ``1/(1+λ)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: int = Field(ge=0, le=MAX_LEVEL)
    name: str
    trigger: str
    response: str
    as_of: date | None
    n_danger: int
    n_confirmed: int
    dials: list[DialAssessment]
    missing: list[str]
    ignored: list[str]
    spec_version: str
    policy_version: str
    lam: float
    threshold: float


class Governor:
    """Stateful aggregator; call :meth:`evaluate` once per weekly batch of readings."""

    def __init__(
        self,
        spec: TripwireSpec | None = None,
        policy: PolicySpec | None = None,
        *,
        allow_normal_fallback: bool = False,
    ) -> None:
        """Run init."""
        self.spec = spec or load_tripwire_spec()
        self.policy = policy or load_policy()
        self.allow_normal_fallback = allow_normal_fallback
        rules = danger_rules(self.policy.section("danger"))
        missing = [d for d in self.spec.dials if d not in rules]
        if missing:
            raise ValueError(f"policy.yaml has no danger rule for {missing}")
        self.rules: dict[str, DangerRule] = {d: rules[d] for d in self.spec.dials}
        self.min_weight = float(self.policy.section("rawls").get("min_weight", 0.0))
        h = self.policy.jonas.hysteresis()
        self.machines = {d: DialStateMachine(h) for d in self.spec.dials}
        self._last: dict[str, DialAssessment] = {}

    def _assess(self, reading: DialReading) -> DialAssessment:
        rule = self.rules[reading.dial]
        worst_scenario: str | None = None
        method: Method
        if reading.detail.get(SCENARIOS_KEY):
            ens = worst_case_p_danger(reading, rule, min_weight=self.min_weight)
            p, p_exp, worst_scenario, method = (
                ens.worst,
                ens.expected,
                ens.worst_scenario,
                "samples",
            )
            if reading.detail.get(SAMPLES_KEY):
                p = max(p, p_danger(reading, rule).p)
        else:
            pd_ = p_danger(reading, rule, allow_normal_fallback=self.allow_normal_fallback)
            p, p_exp, method = pd_.p, pd_.p, pd_.method
        machine = self.machines[reading.dial].step(p)
        self.machines[reading.dial] = machine
        raw = reading.detail.get(self.spec.confirmation.detail_key)
        groups = None if raw is None else int(raw)
        in_danger = machine.state is DialState.DANGER
        confirmed = (
            in_danger
            and groups is not None
            and groups >= self.spec.confirmation.min_independent_groups
        )
        return DialAssessment(
            dial=reading.dial,
            p_danger=p,
            p_expected=p_exp,
            method=method,
            worst_scenario=worst_scenario,
            alarm=self.policy.jonas.alarm(p),
            state=machine.state,
            confirmed=confirmed,
            n_independent_groups=groups,
            as_of=reading.as_of,
        )

    def _held(self, dial: str) -> DialAssessment:
        last = self._last.get(dial)
        if last is not None:
            return last
        return DialAssessment(
            dial=dial,
            p_danger=0.0,
            p_expected=0.0,
            method=None,
            alarm=False,
            state=self.machines[dial].state,
            confirmed=False,
        )

    def evaluate(self, readings: Iterable[DialReading]) -> TripwireAlert:
        """Assess one batch of readings (at most one per dial) and return the alert."""
        batch = list(readings)
        seen: dict[str, DialReading] = {}
        ignored: list[str] = []
        for r in batch:
            if r.dial not in self.rules:
                ignored.append(r.dial)
            elif r.dial in seen:
                raise ValueError(f"two readings for dial {r.dial!r} in one batch")
            else:
                seen[r.dial] = r
        assessments: list[DialAssessment] = []
        missing: list[str] = []
        for dial in self.spec.dials:
            if dial in seen:
                a = self._assess(seen[dial])
                self._last[dial] = a
            else:
                missing.append(dial)
                a = self._held(dial)
            assessments.append(a)
        n_danger = sum(a.in_danger for a in assessments)
        n_confirmed = sum(a.confirmed for a in assessments)
        lv = self.spec.level(level_for(n_danger, n_confirmed))
        dates = [r.as_of for r in batch]
        alert = TripwireAlert(
            level=lv.level,
            name=lv.name,
            trigger=lv.trigger,
            response=lv.response,
            as_of=max(dates) if dates else None,
            n_danger=n_danger,
            n_confirmed=n_confirmed,
            dials=assessments,
            missing=missing,
            ignored=sorted(set(ignored)),
            spec_version=self.spec.version,
            policy_version=self.policy.version,
            lam=self.policy.jonas.lam,
            threshold=self.policy.jonas.threshold,
        )
        log.info("tripwire level %d (%s): %d dial(s) in danger", lv.level, lv.name, n_danger)
        return alert
