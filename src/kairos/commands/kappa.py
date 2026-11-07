"""``kairos kappa``: estimate the recursion dial ``κ``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(help="Recursion dial κ (NeuralProphet paired models).", no_args_is_help=True)


def _emit(design: pd.DataFrame, seeds: int, permutations: int, epochs: int, n_jobs: int) -> None:
    from kairos.neural.kappa import estimate_kappa, kappa_reading
    from kairos.neural.paired import PairedSpec

    spec = PairedSpec(config=PairedSpec().config.model_copy(update={"epochs": epochs}))
    result = estimate_kappa(design, spec, n_seeds=seeds, n_permutations=permutations, n_jobs=n_jobs)
    as_of = pd.Timestamp(design["ds"].max()).date()
    reading = kappa_reading(result, as_of)
    typer.echo(json.dumps(reading.model_dump(mode="json"), indent=2))


@app.command()
def synthetic(
    coupling: float = typer.Option(0.8, help="Self-coupling phi of the simulated target."),
    months: int = typer.Option(192, help="Months to simulate."),
    seeds: int = typer.Option(20, help="Seeds per model."),
    permutations: int = typer.Option(49, help="Permutation-null refits."),
    epochs: int = typer.Option(300, help="Maximum epochs per fit."),
    n_jobs: int = typer.Option(-1, help="Worker processes (-1 = all CPUs)."),
    seed: int = typer.Option(0, help="Simulation seed."),
) -> None:
    """Estimate κ on a simulated series with known self-coupling (offline demo)."""
    from kairos.neural import synthetic_series
    from kairos.neural.regressors import design_frame

    design = design_frame(synthetic_series(months, coupling=coupling, seed=seed), "capability")
    _emit(design, seeds, permutations, epochs, n_jobs)


@app.command()
def estimate(
    path: Path = typer.Argument(..., exists=True, help="SeriesFrame as CSV or Parquet."),  # noqa: B008
    target: str = typer.Option(..., help="series_id of the capability target."),
    seeds: int = typer.Option(20, help="Seeds per model."),
    permutations: int = typer.Option(49, help="Permutation-null refits."),
    epochs: int = typer.Option(300, help="Maximum epochs per fit."),
    n_jobs: int = typer.Option(-1, help="Worker processes (-1 = all CPUs)."),
) -> None:
    """Estimate κ for TARGET against the default drivers in a SeriesFrame file."""
    from kairos.neural.regressors import design_frame

    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, parse_dates=["ds"])
    frame["ds"] = pd.to_datetime(frame["ds"]).astype("datetime64[ns]")
    _emit(design_frame(frame, target), seeds, permutations, epochs, n_jobs)


__all__ = ["app"]
