"""On-disk layout of the dataset and a DuckDB catalog over it.

Layout under ``Settings.data_dir`` (default ``data/``)::

    data/
      raw/<source>/<file>            downloaded inputs (git-ignored)
      processed/series/<id>.parquet  one stored series per file (see kairos.data.schema)
      processed/tables/<name>.parquet  auxiliary tables (per-model rows, reports)
      processed/events.parquet       the event calendar
      MANIFEST.json                  SHA-256 of every processed file (kairos.data.manifest)

Each series file carries its :class:`~kairos.types.SeriesMeta` and any extra
method metadata as JSON in the Parquet key-value metadata, so a file is
self-describing. :func:`connect_catalog` exposes everything to SQL as the views
``series``, ``series_meta`` and ``events``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kairos.data.events import validate_events
from kairos.data.schema import validate_series
from kairos.settings import get_settings
from kairos.types import SeriesFrame, SeriesMeta

__all__ = [
    "DataPaths",
    "Dataset",
    "connect_catalog",
    "data_paths",
    "list_series",
    "read_all",
    "read_events",
    "read_extra",
    "read_meta",
    "read_series",
    "read_table",
    "read_table_meta",
    "write_dataset",
    "write_events",
    "write_table",
]

META_KEY: Final = b"kairos.meta"
EXTRA_KEY: Final = b"kairos.extra"


@dataclass(frozen=True, slots=True)
class Dataset:
    """A series with its metadata, as produced by a source and written to disk.

    Attributes:
        meta: Series metadata.
        frame: Stored series (passes :func:`kairos.data.schema.validate_series`).
        extra: JSON-serialisable method metadata (envelope method, query, caveats...).
    """

    meta: SeriesMeta
    frame: SeriesFrame
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Resolved paths of the dataset tree rooted at ``root``."""

    root: Path

    @property
    def raw(self) -> Path:
        """Downloaded inputs."""
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        """Processed outputs covered by the manifest."""
        return self.root / "processed"

    @property
    def series(self) -> Path:
        """One Parquet file per series."""
        return self.processed / "series"

    @property
    def tables(self) -> Path:
        """Auxiliary Parquet tables."""
        return self.processed / "tables"

    @property
    def events(self) -> Path:
        """The event calendar."""
        return self.processed / "events.parquet"

    @property
    def manifest(self) -> Path:
        """The SHA-256 manifest."""
        return self.root / "MANIFEST.json"

    def series_file(self, series_id: str) -> Path:
        """Path of one series file."""
        return self.series / f"{series_id}.parquet"


def data_paths(root: Path | str | None = None) -> DataPaths:
    """Return the dataset paths, rooted at ``root`` or ``Settings.data_dir``."""
    return DataPaths(Path(root) if root is not None else get_settings().data_dir)


def _write_parquet(df: pd.DataFrame, path: Path, metadata: dict[bytes, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    merged = {**(table.schema.metadata or {}), **metadata}
    table = table.replace_schema_metadata(merged)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return path


def write_dataset(dataset: Dataset, root: Path | str | None = None) -> Path:
    """Validate a dataset and write it to ``processed/series/<series_id>.parquet``.

    Args:
        dataset: The series to store; every row's ``series_id`` must match the metadata.
        root: Dataset root (defaults to ``Settings.data_dir``).

    Returns:
        The written path.

    Raises:
        ValueError: If the frame is invalid or mixes series IDs.
    """
    frame = validate_series(dataset.frame)
    ids = set(frame["series_id"].unique())
    if ids != {dataset.meta.series_id}:
        raise ValueError(f"frame series_id {ids} != meta {dataset.meta.series_id!r}")
    meta = {
        META_KEY: json.dumps(dataclasses.asdict(dataset.meta), sort_keys=True).encode(),
        EXTRA_KEY: json.dumps(dataset.extra, sort_keys=True, default=str).encode(),
    }
    return _write_parquet(frame, data_paths(root).series_file(dataset.meta.series_id), meta)


def _kv(path: Path, key: bytes) -> dict[str, Any]:
    md = pq.read_schema(path).metadata or {}
    if key not in md:
        raise KeyError(f"{path} has no {key.decode()} metadata")
    value: dict[str, Any] = json.loads(md[key])
    return value


def _existing(series_id: str, root: Path | str | None) -> Path:
    path = data_paths(root).series_file(series_id)
    if not path.is_file():
        raise FileNotFoundError(f"no processed series {series_id!r} at {path}")
    return path


def read_series(series_id: str, root: Path | str | None = None) -> SeriesFrame:
    """Read one stored series."""
    return pd.read_parquet(_existing(series_id, root))


def read_meta(series_id: str, root: Path | str | None = None) -> SeriesMeta:
    """Read the :class:`~kairos.types.SeriesMeta` stored with a series."""
    return SeriesMeta(**_kv(_existing(series_id, root), META_KEY))


def read_extra(series_id: str, root: Path | str | None = None) -> dict[str, Any]:
    """Read the extra method metadata stored with a series."""
    return _kv(_existing(series_id, root), EXTRA_KEY)


def list_series(root: Path | str | None = None) -> list[str]:
    """Return the IDs of every stored series, sorted."""
    folder = data_paths(root).series
    return sorted(p.stem for p in folder.glob("*.parquet")) if folder.is_dir() else []


def read_all(root: Path | str | None = None, *, side: str | None = None) -> dict[str, Dataset]:
    """Read every stored series (optionally of one ``side``) as datasets keyed by ID."""
    out: dict[str, Dataset] = {}
    for sid in list_series(root):
        meta = read_meta(sid, root)
        if side is None or meta.side == side:
            out[sid] = Dataset(meta, read_series(sid, root), read_extra(sid, root))
    return out


def write_table(
    df: pd.DataFrame,
    name: str,
    root: Path | str | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write an auxiliary table to ``processed/tables/<name>.parquet``.

    Args:
        df: Table to write.
        name: Table name (file stem).
        root: Dataset root (defaults to ``Settings.data_dir``).
        meta: JSON-serialisable provenance (licence, URL, method) stored in the file.
    """
    blob = json.dumps(meta or {}, sort_keys=True, default=str).encode()
    return _write_parquet(df, data_paths(root).tables / f"{name}.parquet", {EXTRA_KEY: blob})


def read_table(name: str, root: Path | str | None = None) -> pd.DataFrame:
    """Read an auxiliary table written by :func:`write_table`."""
    return pd.read_parquet(data_paths(root).tables / f"{name}.parquet")


def read_table_meta(name: str, root: Path | str | None = None) -> dict[str, Any]:
    """Read the provenance metadata stored with an auxiliary table."""
    return _kv(data_paths(root).tables / f"{name}.parquet", EXTRA_KEY)


def write_events(events: pd.DataFrame, root: Path | str | None = None) -> Path:
    """Validate and write the event calendar to ``processed/events.parquet``."""
    return _write_parquet(validate_events(events), data_paths(root).events, {})


def read_events(root: Path | str | None = None) -> pd.DataFrame:
    """Read the event calendar."""
    return pd.read_parquet(data_paths(root).events)


def connect_catalog(
    root: Path | str | None = None, *, database: str = ":memory:"
) -> duckdb.DuckDBPyConnection:
    """Open DuckDB with catalog views over the processed dataset.

    Views:
        ``series``: every row of every series file (columns unioned by name).
        ``series_meta``: one row per series with its :class:`SeriesMeta` fields.
        ``events``: the event calendar (if written).

    Args:
        root: Dataset root (defaults to ``Settings.data_dir``).
        database: DuckDB database path; in-memory by default.
    """
    paths = data_paths(root)
    con = duckdb.connect(database)
    ids = list_series(root)
    if ids:
        glob = (paths.series / "*.parquet").as_posix()
        con.execute(
            f"CREATE OR REPLACE VIEW series AS "
            f"SELECT * FROM read_parquet('{glob}', union_by_name = true)"
        )
        meta = pd.DataFrame([dataclasses.asdict(read_meta(s, root)) for s in ids])
        con.register("_series_meta_df", meta)
        con.execute("CREATE OR REPLACE VIEW series_meta AS SELECT * FROM _series_meta_df")
    if paths.events.is_file():
        con.execute(
            f"CREATE OR REPLACE VIEW events AS "
            f"SELECT * FROM read_parquet('{paths.events.as_posix()}')"
        )
    return con
