"""``kairos prophet``: Prophet null models over a ``SeriesFrame`` file.

Subcommands:

* ``kairos prophet fit SERIES --out DIR``: fit every series in parallel and write
  ``summary.csv``, ``forecast.csv``, ``changepoints.csv`` and ``run.json`` (wall time).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from kairos.types import SeriesFrame, validate_series_frame

__all__ = ["app", "read_series_frame"]

DEFAULT_OUT = Path("out/prophet")

app = typer.Typer(help="Prophet null models of normal progress.", no_args_is_help=True)


def read_series_frame(path: Path) -> SeriesFrame:
    """Read a ``SeriesFrame`` from Parquet or CSV and validate it."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise typer.BadParameter(f"unsupported file type: {path.suffix} (use .parquet or .csv)")
    frame["ds"] = pd.to_datetime(frame["ds"]).astype("datetime64[ns]")
    frame["y"] = frame["y"].astype("float64")
    frame["series_id"] = frame["series_id"].astype(str)
    return validate_series_frame(frame)


@app.command("fit")
def fit_command(
    series: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="SeriesFrame file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = DEFAULT_OUT,
    n_jobs: Annotated[int | None, typer.Option("--n-jobs", "-j", help="Workers.")] = None,
    horizon: Annotated[int | None, typer.Option(help="Forecast months.")] = None,
    seed: Annotated[int | None, typer.Option(help="Master seed.")] = None,
    cache: Annotated[bool, typer.Option(help="Reuse unchanged fits from the fit cache.")] = False,
) -> None:
    """Fit the null model to every series and write the summary and forecasts."""
    from kairos.models.changepoints import changepoint_table
    from kairos.models.fit_all import fit_all, forecast_frame

    frame = read_series_frame(series)
    result = fit_all(frame, n_jobs=n_jobs, horizon=horizon, seed=seed, cache=cache, desc="fit")
    out.mkdir(parents=True, exist_ok=True)
    result.summary().to_csv(out / "summary.csv", index=False)
    forecast_frame(result.fits).to_csv(out / "forecast.csv", index=False)
    changepoint_table(result.fits).to_csv(out / "changepoints.csv", index=False)
    run = {
        "n_series": len(result.fits),
        "wall_seconds": result.wall_seconds,
        "fit_seconds_total": result.fit_seconds_total,
        "n_jobs": result.n_jobs,
        "series_per_task": result.series_per_task,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    Console().print(
        f"fitted {len(result.fits)} series in {result.wall_seconds:.2f} s "
        f"on {result.n_jobs} workers -> {out}"
    )
