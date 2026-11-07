"""Independence grouping of significant metrics (K-5.13, design note D10).

Metrics from the same data source are not independent evidence. Each series carries
a ``source_group`` (``SeriesFrame`` convention, :mod:`kairos.types`); acceleration is
**confirmed** only when metrics with a BH-significant ``δ > 0``
(:func:`kairos.accel.multitest.delta_significance`) span at least ``min_groups``
distinct source groups (default 3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from kairos.types import SeriesFrame

__all__ = ["Confirmation", "confirm_by_source_group", "source_groups"]

DEFAULT_MIN_GROUPS = 3


@dataclass(frozen=True)
class Confirmation:
    """Outcome of :func:`confirm_by_source_group`.

    Attributes:
        confirmed: At least ``min_groups`` independent groups are significant.
        n_groups: Number of distinct significant source groups.
        min_groups: Threshold used.
        groups: Significant group → its significant series IDs.
    """

    confirmed: bool
    n_groups: int
    min_groups: int
    groups: dict[str, list[str]] = field(default_factory=dict)


def source_groups(frame: SeriesFrame) -> dict[str, str]:
    """Map ``series_id`` → ``source_group`` from a ``SeriesFrame``.

    Falls back to ``source`` and then to the series ID itself when a column is
    missing, so a series is never silently merged with another.

    Raises:
        ValueError: If one series carries more than one group.
    """
    col = next((c for c in ("source_group", "source") if c in frame.columns), None)
    if col is None:
        return {str(s): str(s) for s in frame["series_id"].unique()}
    pairs = frame[["series_id", col]].drop_duplicates()
    if pairs["series_id"].duplicated().any():
        raise ValueError(f"a series has more than one {col}")
    return {
        str(s): (str(g) if pd.notna(g) else str(s))
        for s, g in zip(pairs["series_id"], pairs[col], strict=True)
    }


def confirm_by_source_group(
    significance: pd.DataFrame,
    groups: Mapping[str, str] | SeriesFrame,
    *,
    min_groups: int = DEFAULT_MIN_GROUPS,
) -> Confirmation:
    """Count distinct source groups among significant metrics.

    Args:
        significance: Output of :func:`kairos.accel.multitest.delta_significance`.
        groups: ``series_id → source_group`` mapping or a ``SeriesFrame``.
        min_groups: Groups needed for confirmation.
    """
    if min_groups < 1:
        raise ValueError("min_groups must be >= 1")
    mapping = source_groups(groups) if isinstance(groups, pd.DataFrame) else dict(groups)
    sig = significance.loc[significance["significant"].astype(bool), "series_id"]
    by_group: dict[str, list[str]] = {}
    for sid in sig:
        if sid not in mapping:
            raise KeyError(f"no source_group for series {sid!r}")
        by_group.setdefault(mapping[sid], []).append(str(sid))
    n = len(by_group)
    return Confirmation(
        confirmed=n >= min_groups, n_groups=n, min_groups=min_groups, groups=by_group
    )
