"""Plotly figure builders. Pure functions of the snapshot, testable headlessly."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kairos.dashboard.components import resolved_level
from kairos.dashboard.snapshot import (
    AccelerationFit,
    DialReading,
    LeadTime,
    Snapshot,
    VingeHorizon,
)
from kairos.dashboard.theme import (
    CAPABILITY_COLOR,
    GREY,
    HUMAN_COLOR,
    INK,
    ORANGE,
    SKY,
    STATE_STYLE,
    VERMILLION,
)

FONT = {"family": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "color": INK}


def base_layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    """Apply the shared accessible layout to a figure."""
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


def capability_vs_human(snap: Snapshot) -> go.Figure:
    """C(t) and H(t) together, always (§13), with the gap and the agreed response.

    The function takes the whole snapshot rather than one series so there is
    no way to draw C without H. Both are base-year normalised, so only the
    change in the gap means anything (D8).
    """
    c, h = snap.capability, snap.human_capacity
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06,
        subplot_titles=("Capability C(t) and human capacity H(t)",
                        "Jonas Gap G(t) = ln C - ln H, relative to the base year"),
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=list(c.t), y=list(c.value), name=c.name, mode="lines",
                   line={"color": CAPABILITY_COLOR, "width": 3}),
        row=1, col=1,
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=list(h.t), y=list(h.value), name=h.name, mode="lines",
                   line={"color": HUMAN_COLOR, "width": 3, "dash": "dash"}),
        row=1, col=1,
    )  # fmt: skip
    hv = dict(zip(h.t, h.value, strict=True))
    gt = [t for t in c.t if t in hv]
    gap = [cv - hv[t] for t, cv in zip(c.t, c.value, strict=True) if t in hv]
    fig.add_trace(
        go.Scatter(x=gt, y=gap, name="Gap C - H", mode="lines", fill="tozeroy",
                   line={"color": GREY, "width": 2}, fillcolor="rgba(89,89,89,0.15)"),
        row=2, col=1,
    )  # fmt: skip
    spec = resolved_level(snap.alert)
    base_layout(fig, f"Level {spec.number} ({spec.name}): {spec.response}", height=560)
    fig.update_layout(title={"font": {"size": 14}})
    fig.update_yaxes(title_text=c.unit or "index", row=1, col=1)
    fig.update_yaxes(title_text="gap", row=2, col=1)
    return fig


def _quantile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default method)."""
    pos = (len(sorted_xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def delta_summary(samples: tuple[float, ...]) -> dict[str, float]:
    """Median, 90% interval and P(δ > 0) of the bootstrap distribution."""
    if not samples:
        raise ValueError("no bootstrap samples")
    xs = sorted(samples)
    return {
        "median": _quantile(xs, 0.5),
        "lower": _quantile(xs, 0.05),
        "upper": _quantile(xs, 0.95),
        "p_positive": sum(x > 0 for x in xs) / len(xs),
    }


def acceleration_fit(a: AccelerationFit) -> go.Figure:
    """Observed series, Prophet null fit with interval, and its changepoints."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=list(a.t) + list(a.t)[::-1], y=list(a.upper) + list(a.lower)[::-1],
                   fill="toself", fillcolor="rgba(0,114,178,0.18)", line={"width": 0},
                   name="fit interval", hoverinfo="skip"),
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=list(a.t), y=list(a.fit), name="Prophet null fit", mode="lines",
                   line={"color": HUMAN_COLOR, "width": 2}),
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=list(a.t), y=list(a.observed), name="observed", mode="markers",
                   marker={"color": INK, "size": 4, "symbol": "circle"}),
    )  # fmt: skip
    for cp in a.changepoints:
        fig.add_vline(x=cp, line={"color": ORANGE, "dash": "dot", "width": 2})
    if a.changepoints:  # legend entry for the vertical lines
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines", name="changepoint",
                       line={"color": ORANGE, "dash": "dot", "width": 2}),
        )  # fmt: skip
    base_layout(fig, f"{a.series}: observed, null fit and changepoints", height=420)
    fig.update_yaxes(title_text=a.unit)
    return fig


def delta_distribution(a: AccelerationFit) -> go.Figure:
    """Bootstrap distribution of δ, with δ = 0 (plain exponential) marked."""
    s = delta_summary(a.delta_samples)
    fig = go.Figure(
        go.Histogram(x=list(a.delta_samples), nbinsx=40, name="bootstrap δ",
                     marker={"color": SKY, "line": {"color": INK, "width": 0.5}}),
    )  # fmt: skip
    fig.add_vline(
        x=0,
        line={"color": INK, "width": 2},
        annotation_text="δ = 0 (exponential)",
        annotation_position="top left",
    )
    for key, dash in (("lower", "dash"), ("median", "solid"), ("upper", "dash")):
        fig.add_vline(x=s[key], line={"color": VERMILLION, "dash": dash, "width": 2})
    base_layout(
        fig,
        f"δ bootstrap: median {s['median']:.3f}, 90% interval "
        f"[{s['lower']:.3f}, {s['upper']:.3f}], P(δ > 0) = {s['p_positive']:.0%}",
        height=340,
    )
    fig.update_xaxes(title_text="δ")
    fig.update_yaxes(title_text="resamples")
    fig.update_layout(showlegend=False, bargap=0.02, hovermode="closest")
    return fig


def _decimal_year(iso_date: str) -> float:
    return int(iso_date[:4]) + (int(iso_date[5:7]) - 1) / 12


def linear_trend(t: tuple[str, ...], y: tuple[float, ...]) -> tuple[float, float]:
    """Ordinary least squares ``y = a + b * year``; returns ``(a, b)``."""
    if len(t) < 2:
        raise ValueError("need at least two points for a trend")
    xs = [_decimal_year(d) for d in t]
    mx, my = sum(xs) / len(xs), sum(y) / len(y)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (v - my) for x, v in zip(xs, y, strict=True)) / sxx
    return my - b * mx, b


def vinge_horizon(vh: VingeHorizon, tolerance: float, published: float,
                  trend_months: int = 36) -> go.Figure:  # fmt: skip
    """h*(t) at one error tolerance, with a linear trend over the recent window."""
    y = vh.at(tolerance, published)
    fig = go.Figure(
        go.Scatter(x=list(vh.t), y=list(y), mode="lines", name=f"h* (within {tolerance:g}x)",
                   line={"color": HUMAN_COLOR, "width": 2}),
    )  # fmt: skip
    title = f"Vinge Horizon h*(t): how far ahead forecasts stay within {tolerance:g}x"
    n = min(trend_months, len(vh.t))
    if n >= 2:
        tt, yy = vh.t[-n:], y[-n:]
        a, b = linear_trend(tt, yy)
        fig.add_trace(
            go.Scatter(x=[tt[0], tt[-1]],
                       y=[a + b * _decimal_year(tt[0]), a + b * _decimal_year(tt[-1])],
                       mode="lines", name=f"trend, last {n} months",
                       line={"color": VERMILLION, "width": 3, "dash": "dash"}),
        )  # fmt: skip
        verb = "shrinking" if b < 0 else "growing"
        title += (
            f"<br><sup>{verb} by {abs(b):.1f} {vh.unit} per year over the last {n} months</sup>"
        )
    if tolerance != published:
        title += f"<br><sup>published tolerance is {published:g}x</sup>"
    base_layout(fig, title, height=420)
    fig.update_yaxes(title_text=f"h* ({vh.unit})", rangemode="tozero")
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
