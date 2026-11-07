"""Factory for sources backed by one curated CSV in the repository.

The CSV has ``#`` provenance comments, a ``series_id`` column (when it holds more
than one series), a date column, a value column and optional bound columns. Every
row should carry a ``source_url``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kairos.data.schema import make_series
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.sources.metr import read_curated_csv
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = ["curated_source"]


def curated_source(
    *,
    name: str,
    title: str,
    file: str,
    metas: Mapping[str, SeriesMeta],
    extra: Mapping[str, dict[str, Any]] | None = None,
    date_col: str = "date",
    value_col: str = "value",
    lo_col: str | None = "lo",
    hi_col: str | None = "hi",
) -> Source:
    """Create and register a source that turns one curated CSV into datasets.

    Args:
        name: Source name (folder under ``snapshots/``).
        title: Human-readable title.
        file: CSV file name.
        metas: ``series_id`` -> metadata; with one entry the CSV needs no
            ``series_id`` column.
        extra: ``series_id`` -> extra metadata (method, caveats, direction).
        date_col: Date column.
        value_col: Value column.
        lo_col: Lower-bound column (``None`` if absent).
        hi_col: Upper-bound column (``None`` if absent).

    Returns:
        The registered :class:`~kairos.data.sources.base.Source`.
    """
    first = next(iter(metas.values()))

    def build(ctx: BuildContext) -> BuildResult:
        table = read_curated_csv(ctx.files[file])
        if "series_id" not in table:
            table["series_id"] = first.series_id
        out: list[Dataset] = []
        for sid, meta in metas.items():
            rows = table[table["series_id"] == sid]
            frame = make_series(
                meta,
                rows[date_col],
                rows[value_col],
                vintage=str(ctx.vintage),
                y_lo=rows[lo_col] if lo_col and lo_col in rows else None,
                y_hi=rows[hi_col] if hi_col and hi_col in rows else None,
            )
            info = {"curated": True, **(extra or {}).get(sid, {})}
            out.append(Dataset(meta, frame, info))
        return BuildResult(datasets=out)

    return register(
        Source(
            name=name,
            title=title,
            license=first.license,
            url=first.url,
            files=(RawFile(file),),
            build=build,
        )
    )
