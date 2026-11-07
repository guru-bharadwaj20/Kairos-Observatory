"""``kairos validate``: run the historical validation cases (Phase 9)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kairos.validation.cases import list_cases, run_case

__all__ = ["app"]

app = typer.Typer(
    help="Historical validation: replay detectors on curves with known endings.",
    no_args_is_help=True,
)


@app.command("list")
def list_command() -> None:
    """List the registered validation cases."""
    for name in list_cases():
        typer.echo(name)


@app.command("run")
def run_command(
    case: str = typer.Argument(..., help="Case name (see `kairos validate list`)."),
    as_json: bool = typer.Option(False, "--json", help="Print the summary as JSON."),
) -> None:
    """Run one case and print its summary; exit 1 if it fails its criterion."""
    try:
        result = run_case(case)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    if as_json:
        payload = {"case": result.case, "passed": result.passed, **result.summary}
        typer.echo(json.dumps(payload, default=str, indent=2))
    else:
        status = "SKIP" if result.skipped else ("PASS" if result.passed else "FAIL")
        typer.echo(f"{result.case}: {status}  ({result.criterion})")
        for key, value in result.summary.items():
            typer.echo(f"  {key}: {value}")
        for note in result.notes:
            typer.echo(f"  note: {note}")
    if not (result.passed or result.skipped):
        raise typer.Exit(1)


_DOCS_OPTION = typer.Option(Path("docs"), help="Where validation.md and figures go.")


@app.command("report")
def report_command(
    docs_dir: Path = _DOCS_OPTION,
) -> None:
    """Run every case and write docs/validation.md with figures."""
    from kairos.validation.report import write_report

    out, results = write_report(docs_dir)
    failed = [r.case for r in results if not (r.passed or r.skipped)]
    typer.echo(f"wrote {out} ({len(results)} cases, failed: {failed or 'none'})")
