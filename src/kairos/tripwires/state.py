"""Calm/danger state machine for one dial, with hysteresis.

A dial reads a scalar *score* each week (for the governor this is ``P(danger)``).
A naive ``score > threshold`` test flickers when the score hovers near the
threshold, which would make tripwire levels jump up and down week to week. The
machine below uses two standard devices against flicker:

* **Hysteresis band.** Enter ``DANGER`` only when ``score >= enter``; leave it only
  when ``score < exit``, with ``exit <= enter``. Scores inside ``[exit, enter)``
  keep the current state.
* **Dwell.** A transition happens only after ``dwell`` consecutive readings agree
  with it (``dwell=1`` switches on the first qualifying reading).

The machine is immutable: :meth:`DialStateMachine.step` returns a new machine.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite

__all__ = ["DialState", "DialStateMachine", "Hysteresis"]


class DialState(StrEnum):
    """State of one dial."""

    CALM = "calm"
    DANGER = "danger"


@dataclass(frozen=True, slots=True)
class Hysteresis:
    """Thresholds of the hysteresis band.

    Attributes:
        enter: Score at or above which a calm dial counts towards ``DANGER``.
        exit: Score below which a dangerous dial counts towards ``CALM``.
        dwell: Consecutive agreeing readings needed before a transition.
    """

    enter: float
    exit: float
    dwell: int = 1

    def __post_init__(self) -> None:
        if not (isfinite(self.enter) and isfinite(self.exit)):
            raise ValueError("enter and exit must be finite")
        if self.exit > self.enter:
            raise ValueError(f"exit ({self.exit}) must be <= enter ({self.enter})")
        if self.dwell < 1:
            raise ValueError("dwell must be >= 1")

    @classmethod
    def around(cls, threshold: float, band: float, dwell: int = 1) -> Hysteresis:
        """Band of total width ``band`` whose upper edge is ``threshold``.

        The upper edge is the published alarm threshold, so the band only ever makes
        the governor slower to *stand down*, never slower to raise the alarm.
        """
        if band < 0:
            raise ValueError("band must be >= 0")
        return cls(enter=threshold, exit=threshold - band, dwell=dwell)


@dataclass(frozen=True, slots=True)
class DialStateMachine:
    """Immutable hysteresis state machine.

    Attributes:
        hysteresis: Band and dwell.
        state: Current state.
        pending: Consecutive readings that argued for leaving ``state``.
    """

    hysteresis: Hysteresis
    state: DialState = DialState.CALM
    pending: int = 0

    def wants_switch(self, score: float) -> bool:
        """Whether ``score`` argues for leaving the current state."""
        if self.state is DialState.CALM:
            return score >= self.hysteresis.enter
        return score < self.hysteresis.exit

    def step(self, score: float) -> DialStateMachine:
        """Feed one reading and return the next machine.

        Raises:
            ValueError: If ``score`` is NaN or infinite.
        """
        if not isfinite(score):
            raise ValueError(f"score must be finite, got {score}")
        if not self.wants_switch(score):
            return replace(self, pending=0)
        pending = self.pending + 1
        if pending >= self.hysteresis.dwell:
            other = DialState.DANGER if self.state is DialState.CALM else DialState.CALM
            return replace(self, state=other, pending=0)
        return replace(self, pending=pending)

    def run(self, scores: Iterable[float]) -> list[DialState]:
        """States after each score, starting from this machine."""
        machine = self
        out: list[DialState] = []
        for s in scores:
            machine = machine.step(s)
            out.append(machine.state)
        return out
