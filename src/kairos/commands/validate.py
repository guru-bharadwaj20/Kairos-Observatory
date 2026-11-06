"""``kairos validate``: run the historical validation cases (Phase 9)."""

from __future__ import annotations

import json

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
        typer.echo(f"{result.case}: {'PASS' if result.passed else 'FAIL'}  ({result.criterion})")
        for key, value in result.summary.items():
            typer.echo(f"  {key}: {value}")
        for note in result.notes:
            typer.echo(f"  note: {note}")
    if not result.passed:
        raise typer.Exit(1)
