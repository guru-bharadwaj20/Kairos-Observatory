"""Parametric intervention-cost curve (Collingridge's control dilemma).

The relative cost of an effective intervention made ``L`` months before a threshold
is crossed is

    ``cost(L) = floor + (1 − floor) · exp(−L / tau)``

so ``cost(0) = 1`` (acting at the crossing), the cost falls monotonically with lead
time, and it never drops below ``floor``. ``tau`` and ``floor`` are public policy
parameters (``policy.yaml``, section ``intervention_cost``), where their assumptions
are written down next to the values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from kairos.philosophy.collingridge import LeadTime

__all__ = ["CostAtLeadTime", "InterventionCostCurve"]


class CostAtLeadTime(BaseModel):
    """Relative intervention cost implied by a lead time and its interval.

    Attributes:
        cost: Cost at the point lead time (``floor`` limit if never crossed).
        lo: Cost at the *upper* end of the lead-time interval (cheapest case).
        hi: Cost at the *lower* end of the lead-time interval (dearest case).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cost: float
    lo: float
    hi: float


class InterventionCostCurve(BaseModel):
    """``cost(L) = floor + (1 − floor)·exp(−L/tau)``.

    Attributes:
        tau: Entrenchment time scale in months.
        floor: Cost that remains however early one acts, in ``[0, 1)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tau: float = Field(gt=0)
    floor: float = Field(ge=0, lt=1)

    def cost(self, months: ArrayLike) -> NDArray[np.float64]:
        """Relative cost at lead times ``months`` (``inf`` gives ``floor``)."""
        m = np.asarray(months, dtype=np.float64)
        if (m < 0).any():
            raise ValueError("lead time must be >= 0")
        out: NDArray[np.float64] = self.floor + (1.0 - self.floor) * np.exp(-m / self.tau)
        return out

    def curve(self, max_months: int = 120) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(months, cost)`` on a monthly grid ``0..max_months`` for plotting."""
        months = np.arange(max_months + 1, dtype=np.float64)
        return months, self.cost(months)

    def at(self, lead: LeadTime) -> CostAtLeadTime:
        """Cost implied by a :class:`LeadTime`; censored ends count as infinite."""

        def c(x: float | None) -> float:
            return float(self.cost(np.inf if x is None else x))

        point = None if lead.months is None else float(lead.months)
        # An unknown lower lead-time bound is taken as "now" (cost 1), unless the
        # threshold is never crossed at all.
        dearest = lead.lo if lead.lo is not None or point is None else 0.0
        return CostAtLeadTime(cost=c(point), lo=c(lead.hi), hi=c(dearest))
