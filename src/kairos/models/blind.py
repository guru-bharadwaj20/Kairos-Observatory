"""Blind changepoint check on AI history.

Can the null model find the known regime shifts of AI progress without being
told where they are? The fit uses no event dates. Afterwards:

1. :func:`regime_shifts` merges neighbouring significant changepoints of the same
   sign into *shifts* (Prophet's sparse prior often spreads one break over two
   or three adjacent changepoints) and dates each shift by its ``|delta|``-weighted
   centre.
2. :func:`match_shifts` takes only the **top-k accelerating shifts**, with ``k``
   the number of expected events, and checks each expected event window for a
   shift within ``±tolerance_months``. Unmatched top-k shifts are reported as
   false alarms, so the check cannot pass by proposing many dates.

Expected events (idea §5): deep learning (2012), transformers (2017) and LLM
scaling (2020-23).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from kairos.models.changepoints import DEFAULT_MIN_MAGNITUDE, changepoint_table
from kairos.models.null_model import NullModelFit, ProphetConfig, fit_null_model
from kairos.types import SeriesFrame

__all__ = [
    "AI_HISTORY_SHIFTS",
    "BlindReport",
    "blind_changepoint_check",
    "match_shifts",
    "regime_shifts",
]

AI_HISTORY_SHIFTS: Final[Mapping[str, tuple[str, str]]] = {
    "deep_learning_2012": ("2012-01-01", "2012-12-01"),
    "transformers_2017": ("2017-01-01", "2017-12-01"),
    "llm_scaling_2020_23": ("2020-01-01", "2023-12-01"),
}
"""Known AI regime shifts as inclusive ``(first month, last month)`` windows."""

_MONTH_DAYS: Final = 365.25 / 12


def regime_shifts(
    fit: NullModelFit,
    *,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    merge_months: float = 18.0,
) -> pd.DataFrame:
    """Merge adjacent same-sign significant changepoints into regime shifts.

    Args:
        fit: A fitted null model.
        min_magnitude: Significance threshold on ``|delta|`` (ln units per year).
        merge_months: Neighbours closer than this, with the same sign, form one shift.

    Returns:
        Columns ``series_id, ds, start, end, total_delta, n_changepoints``, ordered by
        ``total_delta`` descending (largest acceleration first).
    """
    table = changepoint_table(fit, min_magnitude=min_magnitude, significant_only=True)
    columns = ["series_id", "ds", "start", "end", "total_delta", "n_changepoints"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    groups: list[list[int]] = [[0]]
    for i in range(1, len(table)):
        prev = table.iloc[groups[-1][-1]]
        row = table.iloc[i]
        gap = (row["ds"] - prev["ds"]).days / _MONTH_DAYS
        if gap <= merge_months and np.sign(row["delta"]) == np.sign(prev["delta"]):
            groups[-1].append(i)
        else:
            groups.append([i])
    rows = []
    for idx in groups:
        block = table.iloc[idx]
        weights = block["magnitude"].to_numpy()
        t = block["ds"].astype("int64").to_numpy().astype("float64")
        centre = pd.Timestamp(int(np.average(t, weights=weights))).to_period("M").to_timestamp()
        rows.append(
            {
                "series_id": fit.series_id,
                "ds": centre,
                "start": block["ds"].iloc[0],
                "end": block["ds"].iloc[-1],
                "total_delta": float(block["delta"].sum()),
                "n_changepoints": len(idx),
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("total_delta", ascending=False, ignore_index=True)


def _months_outside(ds: pd.Timestamp, lo: pd.Timestamp, hi: pd.Timestamp) -> float:
    if lo <= ds <= hi:
        return 0.0
    edge = lo if ds < lo else hi
    return float(abs((ds - edge).days)) / _MONTH_DAYS


def match_shifts(
    shifts: pd.DataFrame,
    expected: Mapping[str, tuple[str, str]] = AI_HISTORY_SHIFTS,
    *,
    tolerance_months: float = 12.0,
    top_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match the top-k accelerating shifts to expected event windows.

    Args:
        shifts: Output of :func:`regime_shifts`.
        expected: ``{event: (first month, last month)}``.
        tolerance_months: Allowed distance outside the window.
        top_k: Shifts considered (default ``len(expected)``).

    Returns:
        ``(matches, false_alarms)``. ``matches`` has one row per event with columns
        ``event, window_start, window_end, found, months_off, hit``; each shift
        matches at most one event. ``false_alarms`` are top-k shifts left unmatched.
    """
    k = len(expected) if top_k is None else top_k
    candidates = shifts.loc[shifts["total_delta"] > 0].head(k).reset_index(drop=True)
    used: set[int] = set()
    rows = []
    for event, (lo_s, hi_s) in expected.items():
        lo, hi = pd.Timestamp(lo_s), pd.Timestamp(hi_s)
        best: tuple[float, int] | None = None
        for i, ds in enumerate(candidates["ds"]):
            if i in used:
                continue
            off = _months_outside(ds, lo, hi)
            if off <= tolerance_months and (best is None or off < best[0]):
                best = (off, i)
        if best is not None:
            used.add(best[1])
        rows.append(
            {
                "event": event,
                "window_start": lo,
                "window_end": hi,
                "found": candidates["ds"].iloc[best[1]] if best else pd.NaT,
                "months_off": best[0] if best else float("nan"),
                "hit": best is not None,
            }
        )
    false_alarms = candidates.drop(index=sorted(used)).reset_index(drop=True)
    return pd.DataFrame(rows), false_alarms


@dataclass(frozen=True)
class BlindReport:
    """Result of :func:`blind_changepoint_check`.

    Attributes:
        series_id: Series examined.
        shifts: All regime shifts found.
        matches: One row per expected event.
        false_alarms: Top-k accelerating shifts matching no event.
    """

    series_id: str
    shifts: pd.DataFrame
    matches: pd.DataFrame
    false_alarms: pd.DataFrame

    @property
    def passed(self) -> bool:
        """Every expected event was found within tolerance."""
        return bool(self.matches["hit"].all())


def blind_changepoint_check(
    frame: SeriesFrame,
    series_id: str | None = None,
    *,
    expected: Mapping[str, tuple[str, str]] = AI_HISTORY_SHIFTS,
    tolerance_months: float = 12.0,
    config: ProphetConfig | None = None,
    seed: int | None = None,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    fit: NullModelFit | None = None,
) -> BlindReport:
    """Fit the null model without event information and score its changepoints.

    Args:
        frame: Capability series in raw units (e.g. frontier training compute).
        series_id: Series to check (optional for a one-series frame).
        expected: Event windows to recover.
        tolerance_months: Allowed miss outside each window.
        config: Null-model settings.
        seed: Master seed.
        min_magnitude: Significance threshold for changepoints.
        fit: Reuse an existing fit instead of fitting.
    """
    if fit is None:
        fit = fit_null_model(
            frame, series_id, config=config, seed=seed, horizon=0, keep_model=False
        )
    shifts = regime_shifts(fit, min_magnitude=min_magnitude)
    matches, false_alarms = match_shifts(shifts, expected, tolerance_months=tolerance_months)
    return BlindReport(fit.series_id, shifts, matches, false_alarms)
