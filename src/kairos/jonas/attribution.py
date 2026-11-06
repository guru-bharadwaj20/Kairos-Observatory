"""Component attribution of the Jonas Gap: which H sub-series lags most (K-7.04).

This is the funders' view: where would money for human oversight capacity close
the gap fastest? Over a trailing window, each H component ``i`` gets

* ``rate``: its growth ``d ln h_i/dt`` per year (OLS slope, Newey–West ``se``),
* ``lag``: ``g_C − rate``, how far it falls behind the capability composite,
* ``contribution``: ``w_i · lag``, its share of ``dG/dt``. With the weights fixed
  over the window the contributions sum exactly to the slope of ``G``, because the
  OLS slope is linear and the composite is a weighted mean,
* ``loo_dgdt``: ``dG/dt`` over the same window with the component left out.

Rows are ranked by ``lag``, largest first.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
import statsmodels.api as sm

from kairos.jonas.gap import component_log_index, jonas_gap, side_index, split_sides
from kairos.jonas.sustained import newey_west_lags
from kairos.jonas.velocity import decimal_years
from kairos.types import SeriesFrame

__all__ = [
    "ATTRIBUTION_COLUMNS",
    "GrowthRate",
    "component_attribution",
    "growth_rate",
]

ATTRIBUTION_COLUMNS: Final = (
    "rank",
    "series_id",
    "weight",
    "rate",
    "rate_se",
    "lag",
    "contribution",
    "loo_dgdt",
)


@dataclass(frozen=True)
class GrowthRate:
    """Trailing log growth rate per year with a Newey–West standard error."""

    rate: float
    se: float
    n: int


def growth_rate(ln_values: pd.Series, months: int = 36) -> GrowthRate:
    """OLS slope per year of a log series (indexed by ``ds``) over its last ``months`` points.

    Raises:
        ValueError: With fewer than 3 observations in the window.
    """
    s = ln_values.dropna().sort_index().iloc[-months:]
    n = len(s)
    if n < 3:
        raise ValueError("growth_rate needs at least 3 observations")
    t = decimal_years(pd.DatetimeIndex(s.index))
    x = sm.add_constant(t - t.mean())
    fit = sm.OLS(s.to_numpy(dtype="float64"), x).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(n)}
    )
    return GrowthRate(rate=float(fit.params[1]), se=float(fit.bse[1]), n=n)


def _window(index: pd.DatetimeIndex, months: int) -> pd.DatetimeIndex:
    return index.sort_values()[-months:]


def component_attribution(
    frame: SeriesFrame,
    base_year: int,
    *,
    months: int = 36,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Rank H components by how far they lag the capability composite.

    Args:
        frame: ``SeriesFrame`` with ``"C"`` and ``"H"`` ``source_group`` rows.
        base_year: Normalisation year (does not affect rates).
        months: Trailing window, aligned to the last months of ``G``.
        weights: Optional component weights.

    Returns:
        Frame with :data:`ATTRIBUTION_COLUMNS`, one row per H component.
    """
    gap = jonas_gap(frame, base_year, weights)
    window = _window(pd.DatetimeIndex(gap["ds"]), months)
    c_index = side_index(split_sides(frame)["C"], base_year, weights)["ln_index"]
    g_c = growth_rate(c_index.reindex(window), months).rate

    h_logs = component_log_index(split_sides(frame)["H"], base_year).reindex(window)
    ids = [str(c) for c in h_logs.columns]
    w = np.array([float((weights or {}).get(i, 1.0)) for i in ids])
    w = w / w.sum()

    rows = []
    for k, sid in enumerate(ids):
        g = growth_rate(h_logs[sid], months)
        loo = np.nan
        if len(ids) > 1:
            rest = frame.loc[frame["series_id"] != sid]
            gap_loo = jonas_gap(rest, base_year, weights).set_index("ds")["gap"]
            loo = growth_rate(gap_loo.reindex(window), months).rate
        rows.append(
            {
                "series_id": sid,
                "weight": float(w[k]),
                "rate": g.rate,
                "rate_se": g.se,
                "lag": g_c - g.rate,
                "contribution": float(w[k]) * (g_c - g.rate),
                "loo_dgdt": loo,
            }
        )
    out = pd.DataFrame(rows).sort_values("lag", ascending=False, kind="mergesort")
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.loc[:, list(ATTRIBUTION_COLUMNS)].reset_index(drop=True)
