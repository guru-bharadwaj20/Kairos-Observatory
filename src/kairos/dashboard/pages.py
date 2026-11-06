"""Page renderers. Each takes a snapshot and draws it with Streamlit."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from kairos.dashboard.snapshot import Snapshot


def render_provenance(snap: Snapshot) -> None:
    """Run render provenance."""
    if snap.is_demo:
        st.warning(
            "DEMO DATA. This page shows a synthetic snapshot for testing the layout. "
            "Nothing on it is a forecast."
        )
    st.caption(
        f"Snapshot generated {snap.generated_at} · data vintage {snap.data_vintage} · "
        f"tripwires v{snap.parameters.tripwires_version} · λ = {snap.parameters.lam:g}"
    )


def overview(snap: Snapshot) -> None:
    """Run overview."""
    st.header("Overview")
    st.write(f"Alert level {snap.alert.level}.")


PAGES: dict[str, Callable[[Snapshot], None]] = {"Overview": overview}
