"""Closing scenarios: the H growth needed to bring ``dG/dt`` to 0 (K-7.05).

Since ``dG/dt = g_C − g_H``, the gap stops widening when the human-capacity
composite grows as fast as capability: ``g_H* = g_C``. Over a trailing window:

* **Composite scenario.** ``required_rate = g_C`` with a ``level`` interval from the
  Newey–West standard error of ``g_C``; ``shortfall = g_C − g_H`` is the extra growth
  per year H needs, with an interval from the HAC standard error of the slope of
  ``G`` itself (which accounts for the covariance of ``C`` and ``H``).
* **Single-component scenarios.** If only component ``i`` is scaled up and the
  others keep their current rates, it must grow at
  ``(g_C − Σ_{j≠i} w_j g_j) / w_i``. The interval carries the uncertainty in ``g_C``.

Rates are continuous log growth per year; ``*_pct`` columns convert them to annual
percentage growth, ``100·(e^g − 1)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from kairos.jonas.attribution import growth_rate
from kairos.jonas.gap import component_log_index, jonas_gap, side_index, split_sides
from kairos.types import SeriesFrame

__all__ = [
    "SCENARIO_COLUMNS",
    "ClosingScenario",
    "closing_scenarios",
    "to_pct",
]

SCENARIO_COLUMNS: Final = (
    "series_id",
    "weight",
    "current_rate",
    "required_rate",
    "required_lo",
    "required_hi",
    "multiplier",
    "required_pct",
)


def to_pct(rate: float | np.ndarray[Any, Any]) -> Any:
    """Convert a continuous log growth rate per year to annual percent growth."""
    return 100.0 * np.expm1(rate)


@dataclass(frozen=True)
class ClosingScenario:
    """Composite closing scenario plus per-component scenarios.

    Attributes:
        months: Trailing window length.
        level: Two-sided interval coverage.
        g_c: Capability composite growth per year.
        g_h: Human-capacity composite growth per year.
        required_rate: H growth that brings ``dG/dt`` to 0 (``= g_c``).
        required_lo: Lower interval bound of ``required_rate``.
        required_hi: Upper interval bound of ``required_rate``.
        shortfall: ``g_c − g_h``, the extra H growth needed per year.
        shortfall_lo: Lower interval bound of ``shortfall``.
        shortfall_hi: Upper interval bound of ``shortfall``.
        components: Per-H-component scenarios (:data:`SCENARIO_COLUMNS`).
    """

    months: int
    level: float
    g_c: float
    g_h: float
    required_rate: float
    required_lo: float
    required_hi: float
    shortfall: float
    shortfall_lo: float
    shortfall_hi: float
    components: pd.DataFrame

    @property
    def required_pct(self) -> float:
        """``required_rate`` as annual percent growth."""
        return float(to_pct(self.required_rate))

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly ``dict``; components become a list of records."""
        d = asdict(self)
        d["components"] = self.components.to_dict(orient="records")
        d["required_pct"] = self.required_pct
        return d


def closing_scenarios(
    frame: SeriesFrame,
    base_year: int,
    *,
    months: int = 36,
    level: float = 0.90,
    weights: Mapping[str, float] | None = None,
) -> ClosingScenario:
    """Compute the H growth rates that would stop the gap widening.

    Args:
        frame: ``SeriesFrame`` with ``"C"`` and ``"H"`` ``source_group`` rows.
        base_year: Normalisation year (does not affect rates).
        months: Trailing window aligned to the last months of ``G``.
        level: Two-sided coverage of the reported intervals.
        weights: Optional component weights.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    z = float(stats.norm.ppf(0.5 + level / 2))
    gap = jonas_gap(frame, base_year, weights).set_index("ds")
    window = pd.DatetimeIndex(gap.index).sort_values()[-months:]
    sides = split_sides(frame)
    c = growth_rate(side_index(sides["C"], base_year, weights)["ln_index"].reindex(window), months)
    h = growth_rate(gap["ln_h"].reindex(window), months)
    g = growth_rate(gap["gap"].reindex(window), months)

    h_logs = component_log_index(sides["H"], base_year).reindex(window)
    ids = [str(col) for col in h_logs.columns]
    w = np.array([float((weights or {}).get(i, 1.0)) for i in ids])
    w = w / w.sum()
    rates = np.array([growth_rate(h_logs[i], months).rate for i in ids])
    weighted = float(w @ rates)
    rows = []
    for k, sid in enumerate(ids):
        others = weighted - w[k] * rates[k]

        def need(gc: float, k: int = k, others: float = others) -> float:
            return float((gc - others) / w[k])

        required = need(c.rate)
        rows.append(
            {
                "series_id": sid,
                "weight": float(w[k]),
                "current_rate": float(rates[k]),
                "required_rate": required,
                "required_lo": need(c.rate - z * c.se),
                "required_hi": need(c.rate + z * c.se),
                "multiplier": required / rates[k] if rates[k] > 0 else np.inf,
                "required_pct": float(to_pct(required)),
            }
        )
    comps = pd.DataFrame(rows, columns=list(SCENARIO_COLUMNS))
    return ClosingScenario(
        months=months,
        level=level,
        g_c=c.rate,
        g_h=h.rate,
        required_rate=c.rate,
        required_lo=c.rate - z * c.se,
        required_hi=c.rate + z * c.se,
        shortfall=g.rate,
        shortfall_lo=g.rate - z * g.se,
        shortfall_hi=g.rate + z * g.se,
        components=comps.sort_values("required_rate", kind="mergesort").reset_index(drop=True),
    )
