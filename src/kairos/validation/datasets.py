"""Offline snapshots of the historical validation series.

Small, openly licensed snapshots ship inside the package (``validation/data/``)
so the validation suite runs without network access. Each snapshot has a
:class:`~kairos.types.SeriesMeta` entry in :data:`SNAPSHOTS` with its licence and
source URL; the same sources are listed in ``LICENSE-DATA``.
"""

from __future__ import annotations

from importlib import resources

import pandas as pd

from kairos.types import SeriesFrame, SeriesMeta, validate_series_frame

__all__ = ["SNAPSHOTS", "load_snapshot", "read_snapshot_csv"]

SNAPSHOTS: dict[str, tuple[str, SeriesMeta]] = {}
"""``series_id -> (file name, metadata)`` for every shipped snapshot."""


def read_snapshot_csv(filename: str) -> pd.DataFrame:
    """Read a raw snapshot CSV from the package data directory."""
    path = resources.files("kairos.validation").joinpath("data", filename)
    with resources.as_file(path) as file:
        return pd.read_csv(file)


def load_snapshot(series_id: str) -> SeriesFrame:
    """Load a shipped snapshot with ``ds, y`` columns as a validated ``SeriesFrame``."""
    try:
        filename, meta = SNAPSHOTS[series_id]
    except KeyError:
        raise KeyError(f"no snapshot {series_id!r}; known: {sorted(SNAPSHOTS)}") from None
    raw = read_snapshot_csv(filename)
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(raw["ds"]).astype("datetime64[ns]"),
            "y": raw["y"].astype("float64"),
            "series_id": meta.series_id,
            "source": meta.source,
            "source_group": meta.source_group,
            "unit": meta.unit,
        }
    )
    return validate_series_frame(frame)
