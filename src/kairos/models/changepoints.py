"""Changepoint tables from fitted null models.

Prophet places ``n_changepoints`` *potential* changepoints and shrinks most slope
changes towards zero through its Laplace prior. The table lists all of them, with
the slope change ``delta`` converted to natural-log units per year, the growth
rate on either side, and a ``significant`` flag for changes of at least
``min_magnitude`` (default 0.01 ln/yr, about a 1 % change in annual growth).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

import numpy as np
import pandas as pd

from kairos.models.null_model import NullModelFit

__all__ = ["CHANGEPOINT_COLUMNS", "DEFAULT_MIN_MAGNITUDE", "changepoint_table"]

DEFAULT_MIN_MAGNITUDE: Final = 0.01
"""Smallest ``|delta|`` (ln units per year) counted as a real slope change."""

CHANGEPOINT_COLUMNS: Final = (
    "series_id",
    "ds",
    "delta",
    "rate_before",
    "rate_after",
    "magnitude",
    "significant",
    "rank",
)


def _one(fit: NullModelFit, min_magnitude: float) -> pd.DataFrame:
    deltas = np.asarray(fit.deltas, dtype="float64")
    after = fit.base_rate + np.cumsum(deltas)
    before = after - deltas
    magnitude = np.abs(deltas)
    table = pd.DataFrame(
        {
            "series_id": fit.series_id,
            "ds": pd.DatetimeIndex(fit.changepoints).astype("datetime64[ns]"),
            "delta": deltas,
            "rate_before": before,
            "rate_after": after,
            "magnitude": magnitude,
            "significant": magnitude >= min_magnitude,
        }
    )
    table["rank"] = table["magnitude"].rank(ascending=False, method="first").astype("int64")
    return table


def changepoint_table(
    fits: NullModelFit | Mapping[str, NullModelFit] | Iterable[NullModelFit],
    *,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    significant_only: bool = False,
) -> pd.DataFrame:
    """Changepoint dates, slope deltas and magnitudes per series.

    Args:
        fits: One fit, a ``{series_id: fit}`` mapping or an iterable of fits.
        min_magnitude: ``|delta|`` threshold (ln units per year) for ``significant``.
        significant_only: Drop changepoints below the threshold.

    Returns:
        Frame with :data:`CHANGEPOINT_COLUMNS`, ordered by series then date.
        ``rank`` is 1 for the largest ``|delta|`` within a series.
    """
    if isinstance(fits, NullModelFit):
        items: list[NullModelFit] = [fits]
    elif isinstance(fits, Mapping):
        items = list(fits.values())
    else:
        items = list(fits)
    if not items:
        return pd.DataFrame(columns=list(CHANGEPOINT_COLUMNS))
    table = pd.concat([_one(f, min_magnitude) for f in items], ignore_index=True)
    if significant_only:
        table = table.loc[table["significant"]].reset_index(drop=True)
    return table[list(CHANGEPOINT_COLUMNS)]
