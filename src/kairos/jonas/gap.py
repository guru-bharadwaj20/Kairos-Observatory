"""Jonas Gap series ``G(t) = ln C(t) − ln H(t)`` (K-7.01).

``C`` and ``H`` are measured in different units, so the *level* of ``G`` carries no
meaning (design note D8). Every component is therefore normalised to its own mean
over a base year: its log index is 0 there, and each side's composite is the
(weighted) geometric mean of its normalised components. ``G`` is then 0 in the base
year and only its changes, above all ``dG/dt``, are interpretable.

Inputs follow the :mod:`kairos.types` ``SeriesFrame`` convention. The side of each
row is read from ``source_group``: ``"C"`` for capability series and ``"H"`` for
human-capacity series (case-insensitive). A frame may carry the finished composite
indices from K-2.21 (one series per side) or the raw components; the arithmetic is
the same.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal

import numpy as np
import pandas as pd

from kairos.types import SeriesFrame, validate_series_frame

__all__ = [
    "GAP_COLUMNS",
    "GapSide",
    "component_log_index",
    "jonas_gap",
    "side_components",
    "side_index",
    "split_sides",
]

GapSide = Literal["C", "H"]
GAP_COLUMNS: Final = ("ds", "ln_c", "ln_h", "gap", "n_c", "n_h")


def split_sides(frame: SeriesFrame) -> dict[GapSide, SeriesFrame]:
    """Split a ``SeriesFrame`` into its capability and human-capacity parts.

    Args:
        frame: Long-format frame with a ``source_group`` column of ``"C"``/``"H"``.

    Returns:
        ``{"C": ..., "H": ...}``; rows of any other group are dropped.

    Raises:
        ValueError: If the frame is invalid, lacks ``source_group`` or a side is empty.
    """
    validate_series_frame(frame)
    if "source_group" not in frame.columns:
        raise ValueError("Jonas Gap needs a source_group column marking C and H series")
    group = frame["source_group"].astype(str).str.strip().str.upper()
    out: dict[GapSide, SeriesFrame] = {}
    for side in ("C", "H"):
        part = frame.loc[group == side]
        if part.empty:
            raise ValueError(f"no {side} series found (source_group == {side!r})")
        out[side] = part
    return out


def side_components(frame: SeriesFrame) -> dict[GapSide, list[str]]:
    """Return the sorted component ``series_id`` values on each side."""
    sides = split_sides(frame)
    return {side: sorted(part["series_id"].astype(str).unique()) for side, part in sides.items()}


def component_log_index(part: SeriesFrame, base_year: int) -> pd.DataFrame:
    """Return ``ln y − mean(ln y over base_year)`` as a wide frame (``ds`` × ``series_id``).

    Raises:
        ValueError: If a component has no observation in ``base_year``.
    """
    wide = part.pivot_table(
        index="ds", columns="series_id", values="y", aggfunc="first", observed=True
    ).sort_index()
    logs = np.log(wide.astype("float64"))
    in_base = logs.index.year == base_year
    base = logs.loc[in_base].mean(axis=0, skipna=True)
    missing = sorted(str(c) for c in base.index[base.isna()])
    if missing:
        raise ValueError(f"components without data in base year {base_year}: {missing}")
    return logs - base


def side_index(
    part: SeriesFrame,
    base_year: int,
    weights: Mapping[str, float] | None = None,
    *,
    min_components: int = 1,
) -> pd.DataFrame:
    """Composite log index of one side: weighted mean of normalised component logs.

    Weights are renormalised each month over the components observed that month,
    so a missing month in one component does not drag the index.

    Args:
        part: Rows of one side.
        base_year: Calendar year at which every component is normalised to 0.
        weights: Optional ``series_id → weight`` (missing ids weigh 1).
        min_components: Months with fewer observed components are dropped.

    Returns:
        Frame indexed by ``ds`` with columns ``ln_index`` and ``n`` (components used).
    """
    logs = component_log_index(part, base_year)
    w = pd.Series(
        [float((weights or {}).get(str(c), 1.0)) for c in logs.columns], index=logs.columns
    )
    if (w < 0).any() or not (w > 0).any():
        raise ValueError("weights must be non-negative with at least one positive")
    present = logs.notna()
    wsum = present.mul(w, axis=1).sum(axis=1)
    num = logs.fillna(0.0).mul(w, axis=1).sum(axis=1)
    out = pd.DataFrame({"ln_index": num / wsum.where(wsum > 0), "n": present.sum(axis=1)})
    return out.loc[(out["n"] >= min_components) & out["ln_index"].notna()]


def jonas_gap(
    frame: SeriesFrame,
    base_year: int,
    weights: Mapping[str, float] | None = None,
    *,
    min_components: int = 1,
) -> pd.DataFrame:
    """Compute the Jonas Gap ``G(t) = ln C(t) − ln H(t)``, relative to ``base_year``.

    The level of ``G`` is relative by construction (D8): ``G`` averages 0 in the base
    year, and a value of 0.7 means "the capability index has grown by a factor of
    e^0.7 ≈ 2 more than the human-capacity index since the base year", nothing more.

    Args:
        frame: ``SeriesFrame`` with ``source_group`` ``"C"`` and ``"H"`` rows.
        base_year: Normalisation year; every component needs data in it.
        weights: Optional component weights, shared by both sides.
        min_components: Minimum components per side for a month to be kept.

    Returns:
        Frame with columns ``ds, ln_c, ln_h, gap, n_c, n_h`` on the months both
        sides cover, sorted by ``ds``.
    """
    sides = split_sides(frame)
    c = side_index(sides["C"], base_year, weights, min_components=min_components)
    h = side_index(sides["H"], base_year, weights, min_components=min_components)
    joined = c.join(h, how="inner", lsuffix="_c", rsuffix="_h")
    if joined.empty:
        raise ValueError("C and H indices share no months")
    out = pd.DataFrame(
        {
            "ds": joined.index,
            "ln_c": joined["ln_index_c"].to_numpy(),
            "ln_h": joined["ln_index_h"].to_numpy(),
            "n_c": joined["n_c"].to_numpy(dtype="int64"),
            "n_h": joined["n_h"].to_numpy(dtype="int64"),
        }
    )
    out["gap"] = out["ln_c"] - out["ln_h"]
    return out.loc[:, list(GAP_COLUMNS)].reset_index(drop=True)
