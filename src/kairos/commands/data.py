"""``kairos data``: refresh, list and verify the open dataset."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Open dataset: refresh, list and verify.", no_args_is_help=True)

_ROOT = typer.Option(None, "--data-dir", help="Dataset root (default: settings data_dir).")
_SOURCE = typer.Option(None, "--source", "-s", help="Only this source (repeatable).")


@app.command()
def refresh(
    offline: bool = typer.Option(False, "--offline", help="Use committed snapshots only."),
    source: list[str] | None = _SOURCE,
    data_dir: Path | None = _ROOT,
) -> None:
    """Fetch, validate, write Parquet and update the manifest."""
    from kairos.data.pipeline import refresh as run
    from kairos.data.quality import QualityError

    try:
        res = run(data_dir, offline=offline, sources=source or None)
    except QualityError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    warnings = len(res.quality)
    typer.echo(
        f"{len(res.series)} series, {len(res.tables)} tables, {res.n_events} events; "
        f"{warnings} quality warnings; manifest {res.digest[:12]}"
    )
    for name, origin in res.origins.items():
        typer.echo(f"  {name}: {origin}")


@app.command("list")
def list_cmd(data_dir: Path | None = _ROOT) -> None:
    """List the stored series with side, unit, range and source."""
    from kairos.data.store import list_series as ids
    from kairos.data.store import read_meta, read_series

    table = Table("series_id", "side", "unit", "from", "to", "n", "source")
    for sid in ids(data_dir):
        m, f = read_meta(sid, data_dir), read_series(sid, data_dir)
        table.add_row(sid, m.side, m.unit, f"{f['ds'].min():%Y-%m}", f"{f['ds'].max():%Y-%m}",
                      str(len(f)), m.source)  # fmt: skip
    Console().print(table)


@app.command()
def verify(data_dir: Path | None = _ROOT) -> None:
    """Check processed files against MANIFEST.json; exit 1 on any mismatch."""
    from kairos.data.manifest import read_manifest, verify_manifest

    problems = verify_manifest(data_dir)
    for p in problems:
        typer.echo(p, err=True)
    if problems:
        raise typer.Exit(1)
    typer.echo(f"manifest ok ({read_manifest(data_dir)['digest'][:12]})")
