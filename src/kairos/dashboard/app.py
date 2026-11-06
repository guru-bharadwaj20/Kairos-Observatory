"""Streamlit entry point. Run through ``kairos dashboard``.

The app reads exactly one published snapshot and renders it; it never fits
models or reads raw data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

from kairos.dashboard.cli import resolve_snapshot
from kairos.dashboard.pages import PAGES, render_provenance
from kairos.dashboard.snapshot import Snapshot, SnapshotError, load_snapshot


def _snapshot_path(argv: list[str]) -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--snapshot", type=Path, default=None)
    known, _ = p.parse_known_args(argv)
    path: Path | None = known.snapshot
    return resolve_snapshot(path)


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> Snapshot:
    del mtime  # cache key only: reload when the file changes
    return load_snapshot(path)


def main() -> None:
    """Run main."""
    st.set_page_config(page_title="KAIROS Observatory", layout="wide")
    path = _snapshot_path(sys.argv[1:])
    try:
        snap = _load(str(path), path.stat().st_mtime)
    except (OSError, SnapshotError) as exc:
        st.error(f"Could not read the published snapshot: {exc}")
        st.stop()
        return
    st.title("KAIROS Observatory")
    render_provenance(snap)
    page = st.sidebar.radio("Page", list(PAGES), key="page")
    PAGES[str(page)](snap)


main()
