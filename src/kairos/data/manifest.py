"""SHA-256 manifest of the processed dataset (``data/MANIFEST.json``).

Format::

    {
      "schema": 1,
      "generated_at": "2026-11-06T04:00:00Z",
      "digest": "<sha256 over the sorted 'path sha256' lines>",
      "files": {"processed/series/c.frontier_compute.parquet": {"sha256": "...", "bytes": 1234}}
    }

Paths are POSIX and relative to the dataset root. ``digest`` depends only on file
contents and paths, never on time, so the ledger can cite one hash for a vintage.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from kairos.data.store import data_paths

__all__ = [
    "MANIFEST_SCHEMA",
    "build_manifest",
    "file_sha256",
    "manifest_digest",
    "read_manifest",
    "verify_manifest",
    "write_manifest",
]

MANIFEST_SCHEMA: Final = 1


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def manifest_digest(files: dict[str, dict[str, Any]]) -> str:
    """Return the content digest of a manifest's ``files`` mapping."""
    lines = "".join(f"{p} {files[p]['sha256']}\n" for p in sorted(files))
    return hashlib.sha256(lines.encode()).hexdigest()


def build_manifest(root: Path | str | None = None) -> dict[str, Any]:
    """Hash every file under ``processed/`` and return the manifest dict."""
    paths = data_paths(root)
    files: dict[str, dict[str, Any]] = {}
    if paths.processed.is_dir():
        for f in sorted(paths.processed.rglob("*")):
            if f.is_file() and f.suffix != ".tmp":
                rel = f.relative_to(paths.root).as_posix()
                files[rel] = {"sha256": file_sha256(f), "bytes": f.stat().st_size}
    return {
        "schema": MANIFEST_SCHEMA,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "digest": manifest_digest(files),
        "files": files,
    }


def write_manifest(root: Path | str | None = None) -> dict[str, Any]:
    """Regenerate ``MANIFEST.json`` and return its content.

    The file is rewritten only if the digest changed, so an unchanged dataset keeps
    its ``generated_at`` stamp (and a clean git diff).
    """
    path = data_paths(root).manifest
    manifest = build_manifest(root)
    if path.is_file():
        old = read_manifest(root)
        if old.get("digest") == manifest["digest"]:
            return old
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return manifest


def read_manifest(root: Path | str | None = None) -> dict[str, Any]:
    """Read ``MANIFEST.json``."""
    value: dict[str, Any] = json.loads(data_paths(root).manifest.read_text(encoding="utf-8"))
    return value


def verify_manifest(root: Path | str | None = None) -> list[str]:
    """Compare files on disk with ``MANIFEST.json``.

    Returns:
        Human-readable problems (``changed``, ``missing`` or ``untracked`` files);
        empty when the dataset matches the manifest.
    """
    recorded = read_manifest(root)["files"]
    current = build_manifest(root)["files"]
    problems = [f"missing: {p}" for p in sorted(set(recorded) - set(current))]
    problems += [f"untracked: {p}" for p in sorted(set(current) - set(recorded))]
    problems += [
        f"changed: {p}"
        for p in sorted(set(recorded) & set(current))
        if recorded[p]["sha256"] != current[p]["sha256"]
    ]
    return problems
