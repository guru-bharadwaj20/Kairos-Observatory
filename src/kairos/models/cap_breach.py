"""Cap-breach detector: observed ``C`` breaking through the ``H``-bounded cap.

Under the logistic null model (:mod:`kairos.models.logistic`) capability cannot
exceed ``cap = ln H(t) + ln(headroom)``. An observation above the cap is data the
"capability is bounded by oversight" world cannot produce, which is itself a
warning (idea §5). Breaches are grouped into episodes of consecutive months.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from kairos.models.logistic import CappedFit

__all__ = ["EPISODE_COLUMNS", "BreachReport", "detect_cap_breaches"]

EPISODE_COLUMNS: Final = ("start", "end", "months", "max_excess", "max_factor")


@dataclass(frozen=True)
class BreachReport:
    """Cap breaches of one capped fit.

    Attributes:
        c_id: Capability series.
        h_id: Human-capacity series.
        excess: ``ln C - cap`` per observed month (positive = above the cap).
        episodes: Runs of at least ``min_months`` breached months, with columns
            :data:`EPISODE_COLUMNS`; ``max_factor = exp(max_excess)``.
        margin: Log-units tolerance used.
    """

    c_id: str
    h_id: str
    excess: pd.Series
    episodes: pd.DataFrame
    margin: float

    @property
    def breached(self) -> bool:
        """At least one breach episode."""
        return not self.episodes.empty

    @property
    def first_breach(self) -> pd.Timestamp | None:
        """Start of the first episode."""
        return None if self.episodes.empty else pd.Timestamp(self.episodes["start"].iloc[0])

    @property
    def ongoing(self) -> bool:
        """The latest observation is inside a breach episode."""
        if self.episodes.empty or self.excess.empty:
            return False
        return bool(self.episodes["end"].iloc[-1] == self.excess.index[-1])

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary for dashboards and ledgers."""
        first = self.first_breach
        return {
            "c_id": self.c_id,
            "h_id": self.h_id,
            "breached": self.breached,
            "ongoing": self.ongoing,
            "n_episodes": len(self.episodes),
            "first_breach": None if first is None else first.strftime("%Y-%m"),
            "max_excess": float(self.excess.max()) if not self.excess.empty else float("nan"),
            "margin": self.margin,
        }


def detect_cap_breaches(
    fit: CappedFit, *, margin: float = 0.0, min_months: int = 1
) -> BreachReport:
    """Report months where observed ``ln C`` exceeds the cap by more than ``margin``.

    Args:
        fit: A :class:`~kairos.models.logistic.CappedFit`.
        margin: Tolerance in log units (``ln 1.1`` ignores breaches under 10 %).
        min_months: Shortest episode reported (filters single noisy months).
    """
    if min_months < 1:
        raise ValueError("min_months must be >= 1")
    observed = fit.forecast.dropna(subset=["y"])
    excess = pd.Series(
        (observed["y"] - observed["cap"]).to_numpy(),
        index=pd.DatetimeIndex(observed["ds"]),
        name="excess",
    )
    above = (excess > margin).to_numpy()
    rows = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(above) and above[j + 1]:
            j += 1
        if j - i + 1 >= min_months:
            block = excess.iloc[i : j + 1]
            peak = float(block.max())
            rows.append(
                {
                    "start": block.index[0],
                    "end": block.index[-1],
                    "months": j - i + 1,
                    "max_excess": peak,
                    "max_factor": float(np.exp(peak)),
                }
            )
        i = j + 1
    episodes = pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))
    return BreachReport(fit.c_id, fit.h_id, excess, episodes, margin)
