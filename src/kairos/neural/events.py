"""Intervention events: releases, laws, export controls, incidents.

An :class:`EventSpec` names one kind of event, the months it happened and the
effect window around each occurrence (``lower_window`` months before,
``upper_window`` months after). :func:`events_frame` turns specs into
NeuralProphet's ``event``/``ds`` table, :func:`add_events` registers them on a
model and :func:`with_events` runs ``create_df_with_events`` so the training
frame carries one indicator column per event.

Phase 2 (K-2.17) owns the curated event calendar; :func:`specs_from_calendar`
accepts any frame with ``event`` and ``ds`` columns plus optional
``lower_window``/``upper_window``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pandas as pd

from kairos.neural.model import quiet_neuralprophet

if TYPE_CHECKING:
    from neuralprophet import NeuralProphet

__all__ = ["EventSpec", "add_events", "events_frame", "specs_from_calendar", "with_events"]


@dataclass(frozen=True, slots=True)
class EventSpec:
    """One kind of event and where it occurred.

    Attributes:
        name: Event name; becomes the NeuralProphet column (no spaces).
        dates: Months of each occurrence (any date in the month is accepted).
        lower_window: Months before each date the effect may start (``<= 0``).
        upper_window: Months after each date the effect may last (``>= 0``).
        mode: ``"additive"`` (log units, i.e. a multiplicative factor on raw values)
            or ``"multiplicative"``.
        regularization: Optional sparsity penalty on the event weights.
    """

    name: str
    dates: tuple[pd.Timestamp, ...]
    lower_window: int = 0
    upper_window: int = 0
    mode: Literal["additive", "multiplicative"] = "additive"
    regularization: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"event name must be alphanumeric/underscore: {self.name!r}")
        if self.lower_window > 0 or self.upper_window < 0:
            raise ValueError("lower_window must be <= 0 and upper_window >= 0")
        if not self.dates:
            raise ValueError(f"event {self.name!r} has no dates")
        months = tuple(sorted({_month_start(d) for d in self.dates}))
        object.__setattr__(self, "dates", months)

    @classmethod
    def of(
        cls,
        name: str,
        dates: Iterable[str | pd.Timestamp],
        *,
        lower_window: int = 0,
        upper_window: int = 0,
        mode: Literal["additive", "multiplicative"] = "additive",
    ) -> EventSpec:
        """Build a spec from date strings or timestamps."""
        return cls(
            name,
            tuple(pd.Timestamp(d) for d in dates),
            lower_window=lower_window,
            upper_window=upper_window,
            mode=mode,
        )


def _month_start(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def events_frame(specs: Sequence[EventSpec]) -> pd.DataFrame:
    """NeuralProphet events table: one row per (event, month)."""
    rows = [(spec.name, d) for spec in specs for d in spec.dates]
    out = pd.DataFrame(rows, columns=["event", "ds"])
    out["ds"] = pd.to_datetime(out["ds"]).astype("datetime64[ns]")
    return out


def specs_from_calendar(calendar: pd.DataFrame) -> list[EventSpec]:
    """Group an event calendar (``event``, ``ds``, optional windows) into specs."""
    missing = {"event", "ds"} - set(calendar.columns)
    if missing:
        raise ValueError(f"calendar is missing columns: {sorted(missing)}")
    specs: list[EventSpec] = []
    for name, group in calendar.groupby("event", sort=True):
        lower = int(group["lower_window"].min()) if "lower_window" in group else 0
        upper = int(group["upper_window"].max()) if "upper_window" in group else 0
        specs.append(EventSpec.of(str(name), group["ds"], lower_window=lower, upper_window=upper))
    return specs


def add_events(model: NeuralProphet, specs: Sequence[EventSpec]) -> NeuralProphet:
    """Register every event (with its window) on ``model``."""
    for spec in specs:
        model.add_events(
            spec.name,
            lower_window=spec.lower_window,
            upper_window=spec.upper_window,
            regularization=spec.regularization,
            mode=spec.mode,
        )
    return model


def with_events(model: NeuralProphet, df: pd.DataFrame, specs: Sequence[EventSpec]) -> pd.DataFrame:
    """Add event indicator columns to ``df`` via ``create_df_with_events``."""
    if not specs:
        return df
    with quiet_neuralprophet():
        out = model.create_df_with_events(df, events_frame(specs))
    return out.reset_index(drop=True)
