"""Source adapter contract, registry and fetch-with-snapshot fallback.

A :class:`Source` names the raw files it needs (each with a live URL, or ``None``
for curated tables maintained in this repository) and a ``build`` function that
turns those files into :class:`~kairos.data.store.Dataset` objects and auxiliary
tables.

Every raw file has a small committed **snapshot** under
``kairos/data/snapshots/<source>/<file>``. :func:`fetch_source` downloads the live
file when it can and falls back to the snapshot when offline or when the download
fails, so the whole pipeline (and every downstream test) runs without a network.
Snapshot retrieval dates live in ``snapshots/VINTAGES.json``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any, Final, Literal

import httpx
import pandas as pd

from kairos.data.store import Dataset
from kairos.logging import get_logger

__all__ = [
    "BuildContext",
    "BuildResult",
    "FetchedSource",
    "RawFile",
    "Source",
    "fetch_source",
    "get_source",
    "register",
    "registered_sources",
    "snapshot_path",
    "snapshot_vintage",
]

log = get_logger(__name__)

USER_AGENT: Final = "kairos-observatory (+https://github.com/guru-bharadwaj20/Kairos-Observatory)"
Origin = Literal["live", "snapshot"]


@dataclass(frozen=True, slots=True)
class RawFile:
    """One input file of a source.

    Attributes:
        name: File name under ``raw/<source>/`` and ``snapshots/<source>/``.
        url: Live download URL; ``None`` for curated files kept in the repository.
        fetch: Custom downloader ``fetch(dest, client)`` for inputs that need more than
            one request (paged APIs); used instead of ``url``.
    """

    name: str
    url: str | None = None
    fetch: Callable[[Path, httpx.Client], None] | None = None

    @property
    def live(self) -> bool:
        """Whether the file can be retrieved from the network."""
        return self.url is not None or self.fetch is not None


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Inputs handed to :attr:`Source.build`.

    Attributes:
        files: Raw file name -> local path.
        vintage: Retrieval date of the inputs.
        upstream: Datasets already built by sources listed in :attr:`Source.requires`.
        tables: Auxiliary tables already built by those sources.
    """

    files: Mapping[str, Path]
    vintage: date
    upstream: Mapping[str, Dataset] = field(default_factory=dict)
    tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Output of :attr:`Source.build`: series plus auxiliary tables and events."""

    datasets: list[Dataset] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    events: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class Source:
    """A data provider adapter.

    Attributes:
        name: Registry key, also the raw/snapshot folder name.
        title: Human-readable provider name.
        license: Licence of the upstream data (matches ``LICENSE-DATA``).
        url: Landing page of the upstream data.
        files: Raw inputs.
        build: Turns fetched inputs into datasets.
        requires: Sources whose outputs ``build`` consumes (built first).
    """

    name: str
    title: str
    license: str
    url: str
    files: tuple[RawFile, ...]
    build: Callable[[BuildContext], BuildResult]
    requires: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FetchedSource:
    """Local inputs of a source after :func:`fetch_source`."""

    source: Source
    files: dict[str, Path]
    origin: dict[str, Origin]
    vintage: date


_REGISTRY: dict[str, Source] = {}

_SOURCE_MODULES: Final = (
    "epoch",
    "metr",
    "benchmarks",
    "costs",
    "research_share",
    "drivers",
    "publications",
    "safety_field",
    "evaluations",
    "regulation",
    "literacy",
    "calendar",
    "validation",
)


def register(source: Source) -> Source:
    """Add a source to the registry (idempotent per name) and return it."""
    _REGISTRY[source.name] = source
    return source


def registered_sources() -> dict[str, Source]:
    """Import every bundled source module and return the registry, in build order."""
    import importlib

    for mod in _SOURCE_MODULES:
        try:
            importlib.import_module(f"kairos.data.sources.{mod}")
        except ModuleNotFoundError as exc:
            if exc.name != f"kairos.data.sources.{mod}":
                raise
    ordered: dict[str, Source] = {}

    def visit(name: str, stack: tuple[str, ...] = ()) -> None:
        if name in ordered:
            return
        if name in stack:
            raise ValueError(f"source dependency cycle: {' -> '.join((*stack, name))}")
        src = _REGISTRY[name]
        for dep in src.requires:
            visit(dep, (*stack, name))
        ordered[name] = src

    for name in sorted(_REGISTRY):
        visit(name)
    return ordered


def get_source(name: str) -> Source:
    """Return a registered source by name."""
    sources = registered_sources()
    if name not in sources:
        raise KeyError(f"unknown source {name!r}; known: {sorted(sources)}")
    return sources[name]


def snapshot_path(source: str, name: str) -> Path:
    """Path of the committed snapshot of ``raw/<source>/<name>``."""
    root = resources.files("kairos.data") / "snapshots" / source / name
    return Path(str(root))


def snapshot_vintage(source: str) -> date:
    """Retrieval date of a source's committed snapshot."""
    vintages: dict[str, Any] = json.loads(
        (resources.files("kairos.data") / "snapshots" / "VINTAGES.json").read_text("utf-8")
    )
    return date.fromisoformat(vintages[source])


def _download(url: str, dest: Path, client: httpx.Client) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
    tmp.replace(dest)


def fetch_source(
    source: Source,
    raw_dir: Path,
    *,
    offline: bool = False,
    client: httpx.Client | None = None,
    today: date | None = None,
) -> FetchedSource:
    """Put every raw input of ``source`` under ``raw_dir/<source>/``.

    Live files are downloaded unless ``offline``; curated files, offline runs and
    failed downloads use the committed snapshot. The vintage is ``today`` when every
    live file was downloaded, otherwise the snapshot's vintage.

    Args:
        source: Source to fetch.
        raw_dir: The ``data/raw`` folder.
        offline: Never touch the network.
        client: HTTP client to use (one is created when ``None``).
        today: Override for the live vintage date (tests).
    """
    folder = raw_dir / source.name
    folder.mkdir(parents=True, exist_ok=True)
    own = client is None and not offline
    http = (
        httpx.Client(timeout=60.0, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        if own
        else client
    )
    files: dict[str, Path] = {}
    origin: dict[str, Origin] = {}
    try:
        for raw in source.files:
            dest = folder / raw.name
            if raw.live and not offline and http is not None:
                try:
                    if raw.fetch is not None:
                        raw.fetch(dest, http)
                    else:
                        assert raw.url is not None
                        _download(raw.url, dest, http)
                    files[raw.name], origin[raw.name] = dest, "live"
                    continue
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    log.warning(
                        "%s: download of %s failed (%s); using snapshot", source.name, raw.name, exc
                    )
            shutil.copyfile(snapshot_path(source.name, raw.name), dest)
            files[raw.name], origin[raw.name] = dest, "snapshot"
    finally:
        if own and http is not None:
            http.close()
    live_urls = [r for r in source.files if r.live]
    all_live = bool(live_urls) and all(origin[r.name] == "live" for r in live_urls)
    vintage = (today or date.today()) if all_live else snapshot_vintage(source.name)
    return FetchedSource(source, files, origin, vintage)
