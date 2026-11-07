"""Policies known in advance, entered as future regressors.

A law with a published effective date is known before it bites, so its
indicator is available for every forecast month. :class:`PolicySpec` encodes it
as a step (``0`` before, ``1`` from the effective month) or a linear phase-in
over ``ramp_months``. NeuralProphet then learns one coefficient per policy and
uses the scheduled values when forecasting.

Unlike events (K-4.03), which are one-off shocks with a short window, a future
regressor is a persistent state; both can be used together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from kairos.neural.events import EventSpec, events_frame
from kairos.neural.model import quiet_neuralprophet

if TYPE_CHECKING:
    from neuralprophet import NeuralProphet

__all__ = [
    "PolicySpec",
    "add_policy_regressors",
    "future_frame",
    "policy_columns",
    "with_policies",
]


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """A scheduled policy.

    Attributes:
        name: Column name of the regressor.
        effective: Month the policy takes effect.
        ramp_months: Months of linear phase-in (``0`` = immediate step).
        until: Optional month the policy lapses (indicator back to 0 from then).
        mode: ``"additive"`` or ``"multiplicative"`` effect in NeuralProphet.
    """

    name: str
    effective: pd.Timestamp
    ramp_months: int = 0
    until: pd.Timestamp | None = None
    mode: Literal["additive", "multiplicative"] = "additive"

    def __post_init__(self) -> None:
        """Run post init."""
        if not self.name.replace("_", "").isalnum():
            raise ValueError(f"policy name must be alphanumeric/underscore: {self.name!r}")
        if self.ramp_months < 0:
            raise ValueError("ramp_months must be >= 0")
        object.__setattr__(self, "effective", pd.Timestamp(self.effective))
        if self.until is not None:
            until = pd.Timestamp(self.until)
            if until <= self.effective:
                raise ValueError("until must be after effective")
            object.__setattr__(self, "until", until)

    def values(
        self, ds: pd.Series | pd.DatetimeIndex
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Indicator in ``[0, 1]`` for each month in ``ds``."""
        stamps = pd.DatetimeIndex(ds)
        months = (stamps.year - self.effective.year) * 12 + (stamps.month - self.effective.month)
        steps = np.asarray(months, dtype=np.float64)
        if self.ramp_months == 0:
            out = (steps >= 0).astype(np.float64)
        else:
            out = np.clip((steps + 1.0) / (self.ramp_months + 1.0), 0.0, 1.0)
        if self.until is not None:
            out[np.asarray(stamps >= self.until)] = 0.0
        return out


def policy_columns(ds: pd.Series | pd.DatetimeIndex, specs: Sequence[PolicySpec]) -> pd.DataFrame:
    """One indicator column per policy for the months in ``ds``."""
    return pd.DataFrame({spec.name: spec.values(ds) for spec in specs})


def with_policies(df: pd.DataFrame, specs: Sequence[PolicySpec]) -> pd.DataFrame:
    """Copy of ``df`` with the policy indicator columns added."""
    out = df.reset_index(drop=True).copy()
    for name, col in policy_columns(out["ds"], specs).items():
        out[name] = col.to_numpy()
    return out


def add_policy_regressors(
    model: NeuralProphet, specs: Sequence[PolicySpec], history: pd.DataFrame
) -> NeuralProphet:
    """Register every policy as a future regressor (no normalisation: it is already 0..1).

    Args:
        model: Unfitted model.
        specs: Policies to register.
        history: Training frame (``ds`` column); used to check identifiability.

    Raises:
        ValueError: If a policy is constant over ``history``: its effect cannot be
            learned (for example a law that only takes effect after the data ends).
    """
    for spec in specs:
        if np.ptp(spec.values(history["ds"])) == 0:
            raise ValueError(
                f"policy {spec.name!r} is constant over the training data; "
                "its effect is not identifiable"
            )
        model.add_future_regressor(spec.name, normalize="off", mode=spec.mode)
    return model


def future_frame(
    model: NeuralProphet,
    history: pd.DataFrame,
    periods: int,
    *,
    policies: Sequence[PolicySpec] = (),
    events: Sequence[EventSpec] = (),
) -> pd.DataFrame:
    """Frame to forecast ``periods`` months past ``history`` with scheduled policies.

    Args:
        model: Fitted model.
        history: The frame the model was fitted on (with policy/event columns).
        periods: Months to forecast.
        policies: Policies registered on the model; their scheduled values are used.
        events: Events registered on the model (future occurrences included).

    Returns:
        Output of ``model.make_future_dataframe`` ready for :func:`kairos.neural.predict`.
    """
    last = pd.Timestamp(history["ds"].max())
    future_ds = pd.date_range(last + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    regressors = policy_columns(future_ds, policies) if policies else None
    with quiet_neuralprophet():
        out = model.make_future_dataframe(
            history,
            events_df=events_frame(events) if events else None,
            regressors_df=regressors,
            periods=periods,
        )
    return out
