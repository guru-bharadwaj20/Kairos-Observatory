"""Page renderers. Each takes a snapshot and draws it with Streamlit."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from kairos.dashboard import components, figures, ledger_view
from kairos.dashboard.snapshot import Snapshot


def render_provenance(snap: Snapshot) -> None:
    """Show where the snapshot came from, and flag demo data."""
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
    """Render the overview page: clock, banner and dials."""
    st.header("KAIROS Clock")
    render_clock(snap)
    st.subheader("The four dials")
    render_dials(snap)


def render_dials(snap: Snapshot) -> None:
    """Render the four dial cards with sparklines."""
    for col, d in zip(st.columns(len(snap.dials)), snap.dials, strict=True):
        with col:
            st.markdown(components.dial_card_html(d), unsafe_allow_html=True)
            st.plotly_chart(figures.sparkline(d), key=f"spark-{d.dial}")


def render_clock(snap: Snapshot) -> None:
    """Render the tripwire banner, level scale and lead-time figure."""
    st.markdown(components.banner_html(snap.alert), unsafe_allow_html=True)
    st.markdown(components.level_scale_html(snap.alert.level), unsafe_allow_html=True)
    if snap.alert.lead_time is not None:
        st.plotly_chart(figures.lead_time_figure(snap.alert.lead_time), key="clock")


def capability_vs_human(snap: Snapshot) -> None:
    """Render the C versus H page."""
    st.header("Capability versus human capacity")
    st.plotly_chart(figures.capability_vs_human(snap), key="c-vs-h")
    st.caption(
        "Both indices are normalised at a base year, so the level of the gap is "
        "arbitrary; only its change over time (dG/dt) carries meaning."
    )


def acceleration(snap: Snapshot) -> None:
    """Render the changepoint and acceleration explorer."""
    st.header("Changepoints and acceleration")
    if not snap.acceleration:
        st.info("This snapshot has no acceleration fits.")
        return
    names = [a.series for a in snap.acceleration]
    choice = st.selectbox("Series", names, key="accel-series")
    a = snap.acceleration[names.index(str(choice))]
    st.plotly_chart(figures.acceleration_fit(a), key="accel-fit")
    st.plotly_chart(figures.delta_distribution(a), key="accel-delta")
    st.caption(
        "δ = 0 is ordinary exponential growth; sustained δ > 0 means growth that "
        "speeds up as the system grows. The interval comes from a moving block "
        "bootstrap (D4), not from a single regression's standard error."
    )


def vinge(snap: Snapshot) -> None:
    """Render the Vinge Horizon page."""
    st.header("Vinge Horizon")
    published = snap.parameters.vinge_tolerance_factor
    options = snap.vinge_horizon.tolerances(published)
    tol = st.select_slider(
        "Error tolerance (forecast within a factor of)",
        options=options,
        value=published,
        format_func=lambda f: f"{f:g}x",
        key="vinge-tol",
    )
    window = st.slider("Trend window (months)", 12, 120, 36, step=6, key="vinge-window")
    st.plotly_chart(
        figures.vinge_horizon(snap.vinge_horizon, float(tol), published, window), key="vinge"
    )
    st.caption(
        "h* is the first forecast horizon at which the smoothed error leaves the "
        "tolerance (D3). Error is measured as a factor in real units (D2). A shrinking "
        "h* means the world is becoming harder to forecast."
    )


def ledger(snap: Snapshot) -> None:
    """Render the ledger browser."""
    st.header("Pre-registration ledger")
    entries = snap.ledger
    counts = ledger_view.proof_counts(entries)
    cols = st.columns(4)
    cols[0].metric("Forecasts", len(entries))
    for col, status in zip(cols[1:], ledger_view.PROOF_LABEL, strict=True):
        col.metric(ledger_view.PROOF_LABEL[status], counts[status])
    breaks = ledger_view.chain_breaks(entries)
    if breaks:
        st.error(f"Hash chain broken at: {', '.join(breaks)}")
    elif entries:
        st.success("Every entry links to the hash of the entry before it.")
    statuses = st.multiselect(
        "Proof status",
        list(ledger_view.PROOF_LABEL.values()),
        default=list(ledger_view.PROOF_LABEL.values()),
        key="ledger-proof",
    )
    rows = [r for r in ledger_view.ledger_rows(entries) if r["Proof"] in statuses]
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Full hash": st.column_config.TextColumn("SHA-256", width="small"),
            "Score": st.column_config.NumberColumn("Score (CRPS, lower is better)",
                                                   format="%.3f"),
        },
    )  # fmt: skip
    st.caption(
        "Entries are append-only and never edited; corrections are new entries. "
        "Run `kairos ledger verify` to recompute every hash and timestamp proof yourself."
    )


PAGES: dict[str, Callable[[Snapshot], None]] = {
    "Overview": overview,
    "C versus H": capability_vs_human,
    "Acceleration explorer": acceleration,
    "Vinge Horizon": vinge,
    "Ledger": ledger,
}
