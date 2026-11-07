"""Offline snapshots of the historical validation series.

Small, openly licensed snapshots ship inside the package (``validation/data/``)
so the validation suite runs without network access. Each snapshot has a
:class:`~kairos.types.SeriesMeta` entry in :data:`SNAPSHOTS` with its licence and
source URL; the same sources are listed in ``LICENSE-DATA``.

Series whose terms do not allow redistribution (index levels and exchange
prices) are **not** shipped: :func:`fetch_fred` downloads them on first use and
caches them under ``Settings.data_dir / "raw" / "validation"`` (not committed).
"""

from __future__ import annotations

import io
from importlib import resources
from pathlib import Path

import pandas as pd

from kairos.logging import get_logger
from kairos.settings import get_settings
from kairos.types import SeriesFrame, SeriesMeta, validate_series_frame

__all__ = [
    "SNAPSHOTS",
    "DataUnavailableError",
    "fetch_fred",
    "load_snapshot",
    "read_snapshot_csv",
]

log = get_logger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class DataUnavailableError(RuntimeError):
    """A runtime-fetched series is neither cached nor downloadable."""


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


def _cache_path(fred_id: str, frequency: str | None) -> Path:
    return (
        get_settings().data_dir
        / "raw"
        / "validation"
        / f"fred_{fred_id}_{frequency or 'Daily'}.csv"
    )


def fetch_fred(
    fred_id: str,
    *,
    series_id: str,
    unit: str,
    frequency: str | None = "Monthly",
    source_group: str = "markets",
    timeout: float = 30.0,
) -> SeriesFrame:
    """Download (or read from cache) a FRED series as a monthly ``SeriesFrame``.

    Args:
        fred_id: FRED series code, e.g. ``"NASDAQCOM"``.
        series_id: KAIROS ``series_id`` to stamp on the rows.
        unit: Unit of the values.
        frequency: FRED aggregation frequency (``"Monthly"`` = monthly average,
            ``None`` = native daily observations; daily frames skip the
            month-start rule of the ``SeriesFrame`` convention).
        source_group: Provenance cluster (D10).
        timeout: HTTP timeout in seconds.

    Raises:
        DataUnavailableError: If the series is not cached and cannot be fetched.
    """
    path = _cache_path(fred_id, frequency)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        import httpx

        try:
            params = {"id": fred_id, **({"fq": frequency} if frequency else {})}
            response = httpx.get(FRED_CSV, params=params, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DataUnavailableError(f"cannot fetch FRED {fred_id}: {exc}") from exc
        text = response.text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        log.info("cached FRED %s at %s", fred_id, path)
    raw = pd.read_csv(io.StringIO(text), na_values=["."])
    raw.columns = ["ds", "y"]
    raw = raw.dropna()
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(raw["ds"]).astype("datetime64[ns]"),
            "y": raw["y"].astype("float64"),
            "series_id": series_id,
            "source": "fred",
            "source_group": source_group,
            "unit": unit,
        }
    ).reset_index(drop=True)
    if frequency is None:
        if frame.duplicated(["series_id", "ds"]).any() or not (frame["y"] > 0).all():
            raise ValueError(f"FRED {fred_id}: duplicate dates or non-positive values")
        return frame
    return validate_series_frame(frame)
