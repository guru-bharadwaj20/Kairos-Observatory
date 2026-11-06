"""Plotly figure builders. Pure functions of the snapshot, testable headlessly."""

from __future__ import annotations

import plotly.graph_objects as go

from kairos.dashboard.snapshot import DialReading, LeadTime
from kairos.dashboard.theme import GREY, INK, STATE_STYLE, VERMILLION

FONT = {"family": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "color": INK}


def base_layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    """Run base layout."""
    fig.update_layout(
        title={"text": title, "x": 0},
        font=FONT,
        height=height,
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E5E5", linecolor=GREY)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5", linecolor=GREY)
    return fig


def lead_time_figure(lead: LeadTime) -> go.Figure:
    """The KAIROS Clock as a number line: median with its interval."""
    pct = f"{lead.interval * 100:.0f}%"
    fig = go.Figure(
        go.Scatter(
            x=[lead.median],
            y=["Lead time"],
            mode="markers",
            marker={"size": 16, "color": VERMILLION, "symbol": "diamond",
                    "line": {"color": INK, "width": 1}},
            error_x={"type": "data", "symmetric": False, "array": [lead.upper - lead.median],
                     "arrayminus": [lead.median - lead.lower], "color": INK, "thickness": 2,
                     "width": 12},
            name=f"median with {pct} interval",
            hovertemplate=(f"median %{{x:.0f}} {lead.unit}<br>{pct} interval "
                           f"{lead.lower:.0f}–{lead.upper:.0f}<extra></extra>"),
        )
    )  # fmt: skip
    base_layout(fig, f"KAIROS Clock: lead time ({pct} interval)", height=180)
    fig.update_xaxes(title=lead.unit, rangemode="tozero")
    fig.update_layout(showlegend=False, hovermode="closest")
    return fig


def sparkline(d: DialReading) -> go.Figure:
    """Recent history of one dial, coloured by its current state."""
    color = STATE_STYLE[d.state][0]
    fig = go.Figure(
        go.Scatter(
            x=list(d.history.t),
            y=list(d.history.value),
            mode="lines",
            line={"color": color, "width": 2},
            name=d.symbol,
            hovertemplate=f"%{{x}}<br>{d.symbol} = %{{y:.3g}}<extra></extra>",
        )
    )
    fig.update_layout(
        height=110,
        margin={"l": 35, "r": 5, "t": 5, "b": 20},
        showlegend=False,
        font=FONT,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(showgrid=False, tickformat="%Y", nticks=3)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", nticks=3)
    return fig
