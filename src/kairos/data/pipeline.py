"""End-to-end dataset refresh: fetch -> build -> validate -> quality -> write -> manifest.

:func:`refresh` is what ``kairos data refresh`` runs. Offline it rebuilds the
whole dataset from the committed snapshots, so the result is reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from kairos.data.composite import build_indices
from kairos.data.events import validate_events
from kairos.data.manifest import write_manifest
from kairos.data.quality import quality_report, raise_on_errors
from kairos.data.schema import validate_series
from kairos.data.sources.base import BuildContext, Source, fetch_source, registered_sources
from kairos.data.store import (
    Dataset,
    data_paths,
    list_series,
    write_dataset,
    write_events,
    write_table,
)
from kairos.logging import get_logger

__all__ = ["RefreshResult", "refresh", "resolve_sources"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RefreshResult:
    """Summary of one refresh.

    Attributes:
        series: IDs of the series written.
        tables: Names of the auxiliary tables written.
        n_events: Rows in the event calendar.
        origins: Source -> ``"live"``, ``"snapshot"`` or ``"mixed"``.
        quality: The quality report (one row per issue).
        digest: Manifest content digest after the refresh.
    """

    series: list[str]
    tables: list[str]
    n_events: int
    origins: dict[str, str]
    quality: pd.DataFrame = field(repr=False)
    digest: str = ""


def resolve_sources(names: Iterable[str] | None = None) -> list[Source]:
    """Return the sources to run, in build order, adding their dependencies."""
    allsrc = registered_sources()
    if names is None:
        return list(allsrc.values())
    wanted: set[str] = set()

    def add(n: str) -> None:
        if n not in allsrc:
            raise KeyError(f"unknown source {n!r}; known: {sorted(allsrc)}")
        wanted.add(n)
        for dep in allsrc[n].requires:
            add(dep)

    for n in names:
        add(n)
    return [s for name, s in allsrc.items() if name in wanted]


def refresh(
    root: Path | str | None = None,
    *,
    offline: bool = False,
    sources: Iterable[str] | None = None,
    client: httpx.Client | None = None,
    today: date | None = None,
) -> RefreshResult:
    """Rebuild the processed dataset and its manifest.

    Args:
        root: Dataset root (defaults to ``Settings.data_dir``).
        offline: Use committed snapshots only.
        sources: Only these sources (and their dependencies). Composite indices are
            rebuilt, and stale series removed, only on a full refresh.
        client: HTTP client for live downloads.
        today: Vintage date for live downloads (tests).

    Raises:
        kairos.data.quality.QualityError: If any series fails an error-level check;
            nothing is written in that case.
    """
    paths = data_paths(root)
    full = sources is None
    datasets: dict[str, Dataset] = {}
    tables: dict[str, pd.DataFrame] = {}
    table_meta: dict[str, dict[str, str]] = {}
    events: list[pd.DataFrame] = []
    origins: dict[str, str] = {}
    vintages: list[date] = []
    for src in resolve_sources(sources):
        fetched = fetch_source(src, paths.raw, offline=offline, client=client, today=today)
        kinds = set(fetched.origin.values())
        origins[src.name] = kinds.pop() if len(kinds) == 1 else "mixed"
        vintages.append(fetched.vintage)
        res = src.build(
            BuildContext(fetched.files, fetched.vintage, upstream=datasets, tables=tables)
        )
        for d in res.datasets:
            validate_series(d.frame)
            datasets[d.meta.series_id] = d
        for name, t in res.tables.items():
            tables[name] = t
            table_meta[name] = {"source": src.name, "license": src.license, "url": src.url}
        if res.events is not None:
            events.append(res.events)
        log.info("%s: %d series (%s)", src.name, len(res.datasets), origins[src.name])

    if full:
        indices, loo = build_indices(datasets, max(vintages))
        datasets.update({d.meta.series_id: d for d in indices})
        for name, t in loo.items():
            tables[name] = t
            table_meta[name] = {"source": "kairos", "license": "CC-BY-4.0", "url": ""}

    report = quality_report(datasets.values())
    raise_on_errors(report)

    if full:
        for stale in set(list_series(root)) - set(datasets):
            paths.series_file(stale).unlink()
    for d in datasets.values():
        write_dataset(d, root)
    for name, t in tables.items():
        write_table(t, name, root, meta=table_meta[name])
    write_table(report, "quality_report", root, meta={"source": "kairos"})
    n_events = 0
    if events:
        ev = validate_events(pd.concat(events, ignore_index=True))
        write_events(ev, root)
        n_events = len(ev)
    manifest = write_manifest(root)
    return RefreshResult(
        series=sorted(datasets),
        tables=sorted([*tables, "quality_report"]),
        n_events=n_events,
        origins=origins,
        quality=report,
        digest=str(manifest["digest"]),
    )
