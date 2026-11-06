"""Page renderers. Each takes a snapshot and draws it with Streamlit."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from kairos.dashboard import components, figures
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
    st.header("KAIROS Clock")
    render_clock(snap)
    st.subheader("The four dials")
    render_dials(snap)


def render_dials(snap: Snapshot) -> None:
    """Run render dials."""
    for col, d in zip(st.columns(len(snap.dials)), snap.dials, strict=True):
        with col:
            st.markdown(components.dial_card_html(d), unsafe_allow_html=True)
            st.plotly_chart(figures.sparkline(d), key=f"spark-{d.dial}")


def render_clock(snap: Snapshot) -> None:
    """Run render clock."""
    st.markdown(components.banner_html(snap.alert), unsafe_allow_html=True)
    st.markdown(components.level_scale_html(snap.alert.level), unsafe_allow_html=True)
    if snap.alert.lead_time is not None:
        st.plotly_chart(figures.lead_time_figure(snap.alert.lead_time), key="clock")


PAGES: dict[str, Callable[[Snapshot], None]] = {"Overview": overview}
