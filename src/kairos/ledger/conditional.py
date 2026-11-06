"""Conditional forecasts: "If no new intervention, X reaches Y by Z (90% interval)".

A forecast about AI can change the behaviour it forecasts (Merton, Jonas). KAIROS
therefore registers claims that are explicitly *conditional* on the set of
intervention events already known at ``as_of`` (Insight 4). Interventions logged
after registration do not make the claim wrong; they make it a counterfactual,
which :mod:`kairos.ledger.scoring` scores as such.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from kairos.ledger.forecast import Forecast

__all__ = [
    "DEFAULT_CONDITION",
    "ConditionalClaim",
    "EventAssumption",
    "crossing_claim",
    "first_crossing",
]

DEFAULT_CONDITION: Final = "no new intervention"


class EventAssumption(BaseModel):
    """An intervention or shock the forecast assumes (an event regressor).

    Attributes:
        name: Event identifier, e.g. ``"eu_ai_act_gpai"``.
        ds: Date the event takes effect.
        kind: ``"intervention"`` (policy), ``"shock"`` (exogenous) or ``"other"``.
        description: Free text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    ds: date
    kind: Literal["intervention", "shock", "other"] = "intervention"
    description: str = ""


class ConditionalClaim(BaseModel):
    """A threshold-crossing claim derived from a quantile forecast.

    "If {condition}, {series_id} reaches {threshold} {unit} by {by} ({coverage:.0%}
    interval {earliest} to {latest})."

    Attributes:
        series_id: Series the claim is about.
        threshold: Level ``Y`` in raw units.
        unit: Unit of ``threshold``.
        by: Median crossing date ``Z`` (first date the median forecast reaches ``Y``),
            or ``None`` if the median does not reach it within the horizon.
        earliest: Crossing date of the upper quantile (lower end of the interval).
        latest: Crossing date of the lower quantile, or ``None`` if it does not
            cross within the horizon ("later than the horizon").
        horizon_end: Last forecast date; ``None`` crossings mean "after this".
        coverage: Nominal coverage of the interval, e.g. ``0.9``.
        condition: Plain-language condition, by default "no new intervention".
        assumed_events: Interventions already known and included in the model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str
    threshold: float
    unit: str = ""
    by: date | None = None
    earliest: date | None = None
    latest: date | None = None
    horizon_end: date
    coverage: float = Field(gt=0, lt=1)
    condition: str = DEFAULT_CONDITION
    assumed_events: tuple[EventAssumption, ...] = ()

    @model_validator(mode="after")
    def _ordered(self) -> ConditionalClaim:
        known = [d for d in (self.earliest, self.by, self.latest) if d is not None]
        if known != sorted(known):
            raise ValueError("crossing dates must satisfy earliest <= by <= latest")
        if self.by is not None and self.earliest is None:
            raise ValueError("the upper quantile must cross no later than the median")
        if self.latest is not None and self.by is None:
            raise ValueError("the median must cross no later than the lower quantile")
        return self

    def statement(self) -> str:
        """Render the claim as one sentence."""
        unit = f" {self.unit}" if self.unit else ""
        y = f"{self.threshold:.4g}{unit}"
        pct = f"{self.coverage:.0%}"
        if self.earliest is None:
            body = (
                f"{self.series_id} does not reach {y} by {self.horizon_end.isoformat()} "
                f"(upper {pct} bound stays below)"
            )
        else:
            by = self.by.isoformat() if self.by else f"after {self.horizon_end.isoformat()}"
            latest = (
                self.latest.isoformat() if self.latest else f"after {self.horizon_end.isoformat()}"
            )
            body = (
                f"{self.series_id} reaches {y} by {by} "
                f"({pct} interval: {self.earliest.isoformat()} to {latest})"
            )
        events = ", ".join(e.name for e in self.assumed_events) or "none"
        return f"If {self.condition}, {body}. Events assumed: {events}."

    def resolved_on(self) -> date:
        """Date after which the claim can be scored (latest bound or horizon end)."""
        return self.latest or self.horizon_end


def first_crossing(forecast: Forecast, level: float, threshold: float) -> date | None:
    """Return the first forecast date whose ``level`` quantile is ``>= threshold``."""
    for point in forecast.points:
        if point.at(level) >= threshold:
            return point.ds
    return None


def crossing_claim(
    forecast: Forecast,
    threshold: float,
    *,
    coverage: float = 0.9,
    assumed_events: tuple[EventAssumption, ...] = (),
    condition: str = DEFAULT_CONDITION,
) -> ConditionalClaim:
    """Derive the conditional claim "``series`` reaches ``threshold`` by Z" from quantiles.

    The interval uses the ``(1 - coverage) / 2`` and ``(1 + coverage) / 2`` quantiles,
    which must be present in every forecast point (``0.05``/``0.95`` for 90%).

    Raises:
        KeyError: If the needed quantile levels are missing.
    """
    lo_level = round((1 - coverage) / 2, 6)
    hi_level = round((1 + coverage) / 2, 6)
    return ConditionalClaim(
        series_id=forecast.series_id,
        threshold=threshold,
        unit=forecast.unit,
        by=first_crossing(forecast, 0.5, threshold),
        earliest=first_crossing(forecast, hi_level, threshold),
        latest=first_crossing(forecast, lo_level, threshold),
        horizon_end=forecast.horizon_end,
        coverage=coverage,
        condition=condition,
        assumed_events=assumed_events,
    )
