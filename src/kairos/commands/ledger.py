"""``kairos ledger``: inspect, verify and timestamp the pre-registration ledger.

* ``kairos ledger verify`` recomputes the hash chain and checks every proof;
  exits 1 on any error.
* ``kairos ledger stamp`` submits unstamped entries to OpenTimestamps calendars
  (needs network and ``uv sync --extra ots``).
* ``kairos ledger upgrade`` completes pending proofs once Bitcoin has confirmed.
* ``kairos ledger scorecard`` prints the calibration and CRPS table.
* ``kairos ledger show`` lists the entries.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kairos.ledger.chain import DEFAULT_LEDGER_PATH, Ledger, LedgerError
from kairos.ledger.scorecard import scorecard, scorecard_markdown
from kairos.ledger.timestamp import (
    DEFAULT_PROOFS_DIR,
    default_timestamper,
    proof_path,
    stamp_entries,
)
from kairos.ledger.verify import report_json, verify_ledger

__all__ = ["app"]

app = typer.Typer(
    help="Pre-registration ledger: verify, timestamp and inspect.", no_args_is_help=True
)

LEDGER_OPT = typer.Option(DEFAULT_LEDGER_PATH, "--ledger", help="Ledger JSONL file.")
PROOFS_OPT = typer.Option(DEFAULT_PROOFS_DIR, "--proofs", help="Directory of .ots proofs.")
OUT_OPT = typer.Option(None, "--out", help="Write Markdown here (else stdout).")


@app.command()
def verify(
    ledger: Path = LEDGER_OPT,
    proofs: Path = PROOFS_OPT,
    require_proofs: bool = typer.Option(
        False, "--require-proofs", help="Fail when an entry has no timestamp proof."
    ),
    expect_head: str | None = typer.Option(
        None, "--expect-head", help="Known-good head entry_hash to check against."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Recompute the hash chain and check every proof; exit 1 on any error."""
    report = verify_ledger(ledger, proofs, require_proofs=require_proofs, expect_head=expect_head)
    if as_json:
        typer.echo(report_json(report))
    else:
        console = Console()
        for issue in report.issues:
            style = "red" if issue.severity == "error" else "yellow"
            console.print(f"[{style}]{issue.severity.upper()}[/] {issue.message}")
        counts: dict[str, int] = {}
        for status in report.proofs.values():
            counts[status.value] = counts.get(status.value, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
        verdict = "[green]OK[/]" if report.ok else f"[red]FAILED ({len(report.errors)} errors)[/]"
        console.print(
            f"{verdict}: {report.n_entries} entries, head {(report.head or '-')[:16]}, "
            f"proofs: {summary}"
        )
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def stamp(ledger: Path = LEDGER_OPT, proofs: Path = PROOFS_OPT) -> None:
    """Timestamp every entry without a proof via OpenTimestamps (uses the network)."""
    try:
        entries = Ledger(ledger).entries()
    except LedgerError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    results = stamp_entries(entries, default_timestamper(), proofs)
    for seq, status in sorted(results.items()):
        typer.echo(f"{seq:6d}  {status.value}")


@app.command()
def upgrade(ledger: Path = LEDGER_OPT, proofs: Path = PROOFS_OPT) -> None:
    """Upgrade pending proofs to Bitcoin-anchored ones (uses the network)."""
    stamper = default_timestamper()
    if stamper is None:
        typer.echo("OpenTimestamps unavailable; run `uv sync --extra ots`", err=True)
        raise typer.Exit(1)
    for entry in Ledger(ledger).entries():
        path = proof_path(entry, proofs)
        if not path.exists():
            continue
        digest = bytes.fromhex(entry.entry_hash)
        new = stamper.upgrade(digest, path.read_bytes())
        path.write_bytes(new)
        typer.echo(f"{entry.seq:6d}  {stamper.check(digest, new).status.value}")


@app.command("scorecard")
def scorecard_cmd(
    ledger: Path = LEDGER_OPT,
    out: Path | None = OUT_OPT,
) -> None:
    """Print (or write) the calibration and CRPS scorecard built from the ledger."""
    text = scorecard_markdown(scorecard(Ledger(ledger)), generated=date.today())
    if out is None:
        typer.echo(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        typer.echo(f"wrote {out}")


@app.command()
def show(ledger: Path = LEDGER_OPT, proofs: Path = PROOFS_OPT) -> None:
    """List the ledger's entries with their proof status."""
    report = verify_ledger(ledger, proofs)
    table = Table("seq", "kind", "recorded_at", "entry_hash", "subject", "proof")
    for entry in Ledger(ledger).entries(check=False):
        subject = ""
        if entry.kind == "forecast":
            f = entry.payload["forecast"]
            subject = f"{f['series_id']} / {f['model']} as of {f['as_of']}"
        elif entry.kind == "intervention":
            subject = str(entry.payload["event"]["name"])
        elif entry.kind in {"correction", "score"}:
            ref = entry.payload.get("supersedes") or entry.payload.get("forecast_entry", "")
            subject = f"-> {str(ref)[:16]}"
        status = report.proofs.get(entry.seq)
        table.add_row(
            str(entry.seq),
            entry.kind,
            entry.recorded_at.isoformat(timespec="seconds"),
            entry.entry_hash[:16],
            subject,
            status.value if status else "-",
        )
    Console().print(table)
