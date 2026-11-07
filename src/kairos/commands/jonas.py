"""``kairos jonas``: read the Jonas Gap dial from a C/H ``SeriesFrame`` file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from kairos.jonas.dial import jonas_dial, load_params
from kairos.types import SeriesFrame, validate_series_frame

__all__ = ["app", "read_series_frame"]

app = typer.Typer(help="Jonas Gap G = ln C - ln H and its velocity dG/dt.", no_args_is_help=True)


def read_series_frame(path: Path) -> SeriesFrame:
    """Read a ``SeriesFrame`` from Parquet or CSV."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["ds"])
    df["ds"] = pd.to_datetime(df["ds"]).astype("datetime64[ns]")
    df["y"] = df["y"].astype("float64")
    return validate_series_frame(df)


@app.command("dial")
def dial(
    path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="SeriesFrame CSV/Parquet.")
    ],
    config: Annotated[Path | None, typer.Option("--config", help="Alternative jonas.yaml.")] = None,
    seed: Annotated[int | None, typer.Option(help="Bootstrap seed (default: settings).")] = None,
    n_jobs: Annotated[int | None, typer.Option("--n-jobs", help="Bootstrap workers.")] = None,
) -> None:
    """Print the Jonas dial reading as JSON."""
    reading = jonas_dial(read_series_frame(path), load_params(config), seed=seed, n_jobs=n_jobs)
    typer.echo(json.dumps(reading.model_dump(mode="json"), indent=2))
