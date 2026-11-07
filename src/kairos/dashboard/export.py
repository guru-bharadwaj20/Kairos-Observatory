"""Static export: one self-contained HTML page for GitHub Pages.

The page carries the same content as the Streamlit app, built from the same
components and figures. Plotly's JavaScript is inlined once, so the page works
offline and loads nothing from third parties. The snapshot it was built from is
copied next to it, so readers can check every number.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

import plotly.graph_objects as go

from kairos.dashboard import components, figures, ledger_view, methodology
from kairos.dashboard.snapshot import Snapshot, load_snapshot

_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1A1A1A;
background:#FFFFFF;margin:0 auto;max-width:1200px;padding:1rem 1.5rem;line-height:1.5}
a{color:#0072B2}h1{margin-bottom:0}nav ul{display:flex;flex-wrap:wrap;gap:1rem;
padding:0;list-style:none}.demo{background:#F0E442;border:2px solid #1A1A1A;
padding:.75rem;font-weight:700}.dials{display:grid;gap:1rem;
grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}table{border-collapse:collapse;
width:100%;font-size:.9rem}th,td{border:1px solid #BDBDBD;padding:.35rem;text-align:left;
vertical-align:top}th{background:#F2F2F2}code{word-break:break-all}
.skip{position:absolute;left:-999px}.skip:focus{left:1rem;top:1rem;background:#FFF}
"""


class _FigureWriter:
    """Embeds figures, inlining plotly.js with the first one only."""

    def __init__(self) -> None:
        self.first = True

    def __call__(self, fig: go.Figure, label: str) -> str:
        div: str = fig.to_html(
            full_html=False,
            include_plotlyjs=self.first,
            config={"displaylogo": False, "responsive": True},
        )
        self.first = False
        return f'<figure aria-label="{html.escape(label)}">{div}</figure>'


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", out)


def markdown_to_html(md: str) -> str:
    """Convert the small Markdown subset used by the methodology page."""
    parts: list[str] = []
    for block in md.split("\n\n"):
        lines = block.splitlines()
        if block.startswith("## "):
            parts.append(f"<h3>{_inline(block[3:])}</h3>")
        elif block.startswith("|") and len(lines) >= 2:
            cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
            head = "".join(f'<th scope="col">{_inline(c)}</th>' for c in cells[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
                for row in cells[2:]
            )
            parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif block.startswith("- "):
            items = "".join(f"<li>{_inline(ln[2:])}</li>" for ln in lines)
            parts.append(f"<ul>{items}</ul>")
        else:
            parts.append(f"<p>{_inline(block)}</p>")
    return "\n".join(parts)


def _ledger_table(snap: Snapshot) -> str:
    rows = ledger_view.ledger_rows(snap.ledger)
    cols = ("ID", "Registered", "Forecast", "Outcome date", "Proof", "Score")
    head = "".join(f'<th scope="col">{c}</th>' for c in (*cols, "SHA-256"))
    body = []
    for r in rows:
        score = r["Score"]
        vals = [html.escape("" if r[c] is None else str(r[c])) for c in cols[:-1]]
        vals.append("" if score is None else f"{float(str(score)):.3f}")
        cells = "".join(f"<td>{v}</td>" for v in vals)
        body.append(f"<tr>{cells}<td><code>{html.escape(str(r['Full hash']))}</code></td></tr>")
    breaks = ledger_view.chain_breaks(snap.ledger)
    status = (
        f"<p><strong>Hash chain broken at: {html.escape(', '.join(breaks))}</strong></p>"
        if breaks
        else "<p>Every entry links to the hash of the entry before it.</p>"
    )
    return f"{status}<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_site(snap: Snapshot, snapshot_href: str = "snapshot.json") -> str:
    """Return the static dashboard as one HTML document."""
    fig = _FigureWriter()
    pub = snap.parameters.vinge_tolerance_factor
    sections: list[tuple[str, str, str]] = []

    clock = components.banner_html(snap.alert) + components.level_scale_html(snap.alert.level)
    if snap.alert.lead_time is not None:
        clock += fig(figures.lead_time_figure(snap.alert.lead_time), "KAIROS Clock lead time")
    dials = "".join(
        f"<div>{components.dial_card_html(d)}{fig(figures.sparkline(d), f'{d.dial} history')}</div>"
        for d in snap.dials
    )
    clock += f'<h3>The four dials</h3><div class="dials">{dials}</div>'
    sections.append(("clock", "KAIROS Clock", clock))
    sections.append(("c-vs-h", "Capability versus human capacity",
                     fig(figures.capability_vs_human(snap), "C(t) and H(t)")))  # fmt: skip
    accel = "".join(
        fig(figures.acceleration_fit(a), f"{a.series} fit")
        + fig(figures.delta_distribution(a), f"{a.series} delta bootstrap")
        for a in snap.acceleration
    )
    sections.append(("acceleration", "Changepoints and acceleration", accel))
    vh = figures.vinge_horizon(snap.vinge_horizon, pub, pub)
    sections.append(("vinge", "Vinge Horizon", fig(vh, "Vinge Horizon")))
    sections.append(("ledger", "Pre-registration ledger", _ledger_table(snap)))
    sections.append(("methodology", "Methodology",
                     markdown_to_html(methodology.methodology_markdown(snap))))  # fmt: skip

    nav = "".join(f'<li><a href="#{sid}">{html.escape(t)}</a></li>' for sid, t, _ in sections)
    body = "".join(
        f'<section id="{sid}" aria-labelledby="{sid}-h"><h2 id="{sid}-h">{html.escape(t)}</h2>'
        f"{content}</section>"
        for sid, t, content in sections
    )
    demo = (
        '<p class="demo" role="alert">DEMO DATA. This page shows a synthetic snapshot for '
        "testing the layout. Nothing on it is a forecast.</p>"
        if snap.is_demo
        else ""
    )
    p = snap.parameters
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>KAIROS Observatory - {html.escape(snap.data_vintage)}</title>"
        f"<style>{_CSS}</style></head><body>"
        '<a class="skip" href="#main">Skip to content</a>'
        "<header><h1>KAIROS Observatory</h1>"
        f"<p>Snapshot generated {html.escape(snap.generated_at)} · data vintage "
        f"{html.escape(snap.data_vintage)} · tripwires v{html.escape(p.tripwires_version)} · "
        f'λ = {p.lam:g} · <a href="{html.escape(snapshot_href)}">snapshot JSON</a></p>'
        f'{demo}<nav aria-label="Sections"><ul>{nav}</ul></nav></header>'
        f'<main id="main">{body}</main>'
        f"<footer><p>{components.DISCLAIMER} Code and data are public; every forecast is "
        "hashed and timestamped before its outcome is known.</p></footer>"
        "</body></html>\n"
    )


def export_site(snapshot_path: str | Path, out_dir: str | Path) -> Path:
    """Write the static site and return the path of index.html."""
    src = Path(snapshot_path)
    snap = load_snapshot(src)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out / "snapshot.json")
    (out / ".nojekyll").write_text("", encoding="utf-8")  # serve files as-is on Pages
    index = out / "index.html"
    index.write_text(render_site(snap), encoding="utf-8")
    return index
