"""Framework-free building blocks: text and HTML fragments.

Both the Streamlit app and the static export use these, so the two always say
the same thing. All snapshot text is HTML-escaped before it is embedded.
"""

from __future__ import annotations

from dataclasses import replace
from html import escape

from kairos.dashboard.snapshot import Alert, DialReading, LeadTime
from kairos.dashboard.theme import DIALS, INK, STATE_STYLE, Level, level

DISCLAIMER = "KAIROS recommends; humans decide."
EN_DASH = "–"


def resolved_level(alert: Alert) -> Level:
    """The level spec, with any name/response the snapshot publishes."""
    spec = level(alert.level)
    if alert.name:
        spec = replace(spec, name=alert.name)
    if alert.response:
        spec = replace(spec, response=alert.response)
    return spec


def _fmt(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 0.05 else f"{x:.1f}"


def lead_time_text(lead: LeadTime | None) -> str:
    """E.g. ``Lead time: 31 months (90% CI: 18-52)``, with an en dash."""
    if lead is None:
        return "Lead time: not estimated (no dial is on course to cross its threshold)"
    pct = _fmt(lead.interval * 100)
    return (
        f"Lead time: {_fmt(lead.median)} {lead.unit} "
        f"({pct}% CI: {_fmt(lead.lower)}{EN_DASH}{_fmt(lead.upper)})"
    )


def banner_html(alert: Alert) -> str:
    """The tripwire banner: level, its trigger and the agreed response."""
    spec = resolved_level(alert)
    return (
        f'<section class="kairos-banner" role="status" aria-live="polite" '
        f'aria-label="Tripwire level {spec.number}: {escape(spec.name)}" '
        f'style="background:{spec.background};color:{spec.foreground};'
        f'padding:1rem 1.25rem;border-radius:0.5rem;border:2px solid #1A1A1A;">'
        f'<div style="font-size:1.6rem;font-weight:700;">'
        f'<span aria-hidden="true">{spec.symbol}</span> '
        f"Level {spec.number}: {escape(spec.name)}</div>"
        f'<div style="font-size:1.25rem;margin-top:0.25rem;">'
        f"{escape(lead_time_text(alert.lead_time))}</div>"
        f'<div style="margin-top:0.5rem;"><strong>Trigger:</strong> {escape(spec.trigger)}</div>'
        f"<div><strong>Recommended response:</strong> {escape(spec.response)}</div>"
        f'<div style="margin-top:0.5rem;font-style:italic;">{DISCLAIMER}</div>'
        f"</section>"
    )


def level_scale_html(current: int) -> str:
    """All five levels in a row, the current one marked, for context."""
    cells = []
    for n in range(5):
        spec = level(n)
        here = n == current
        cells.append(
            f'<li style="flex:1;list-style:none;background:{spec.background};'
            f"color:{spec.foreground};padding:0.35rem;text-align:center;"
            f'{"outline:3px solid #1A1A1A;font-weight:700;" if here else ""}"'
            f"{' aria-current="step"' if here else ''}>"
            f"{n} {escape(spec.name)}</li>"
        )
    return (
        '<ol class="kairos-scale" aria-label="Tripwire levels 0 to 4" '
        'style="display:flex;gap:4px;padding:0;margin:0.5rem 0;">' + "".join(cells) + "</ol>"
    )


def dial_card_html(d: DialReading) -> str:
    """One dial: name, value, state (colour + symbol + words) and P(danger)."""
    info = DIALS[d.dial]
    color, symbol, label = STATE_STYLE[d.state]
    unit = f" {escape(d.unit)}" if d.unit else ""
    return (
        f'<article class="kairos-dial" aria-label="{escape(info.title)} dial: {label}" '
        f'style="border:2px solid {color};border-left:10px solid {color};'
        f'border-radius:0.5rem;padding:0.75rem;">'
        f'<h3 style="margin:0;font-size:1.1rem;">{escape(info.title)}</h3>'
        f'<div style="font-size:1.8rem;font-weight:700;">'
        f"{escape(d.symbol)} = {d.value:.3g}{unit}</div>"
        f'<div style="font-weight:700;color:{INK};">'
        f'<span aria-hidden="true" style="color:{color};">{symbol}</span> {label}</div>'
        f"<div>P(danger) = {d.p_danger:.0%}</div>"
        f'<div style="font-size:0.85rem;color:#3D3D3D;">{escape(info.measures)}. '
        f"Danger sign: {escape(info.danger_sign)}.</div>"
        f"</article>"
    )
