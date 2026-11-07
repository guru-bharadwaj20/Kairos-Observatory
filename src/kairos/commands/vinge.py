"""``kairos vinge``: compute ``h*(t)`` and the Vinge dial from saved backtests.

Input files are Parquet (``.parquet``) or CSV, chosen by extension.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

__all__ = ["app"]

app = typer.Typer(
    help="Vinge Horizon h*: how far ahead can we still forecast?", no_args_is_help=True
)

ThresholdsOpt = Annotated[Path | None, typer.Option(help="Override thresholds YAML.")]


def _read(path: Path, dates: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for col in dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


@app.command("hstar")
def hstar_cmd(
    backtest: Annotated[Path, typer.Argument(exists=True, help="BacktestFrame file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the h*(t) table.")],
    thresholds: ThresholdsOpt = None,
) -> None:
    """Compute h*(t) for every series in a backtest file."""
    from kairos.vinge.dial import load_thresholds, spec_from_thresholds
    from kairos.vinge.series import hstar_series

    bt = _read(backtest, ("cutoff", "ds"))
    table = hstar_series(bt, spec_from_thresholds(load_thresholds(thresholds)))
    if out.suffix == ".parquet":
        table.to_parquet(out, index=False)
    else:
        table.to_csv(out, index=False)
    typer.echo(f"wrote {len(table)} readings to {out}")


@app.command("dial")
def dial_cmd(
    hstar: Annotated[Path, typer.Argument(exists=True, help="h*(t) table from `vinge hstar`.")],
    thresholds: ThresholdsOpt = None,
) -> None:
    """Print the Vinge dial reading as JSON."""
    from kairos.vinge.dial import load_thresholds, vinge_dial

    reading = vinge_dial(_read(hstar, ("as_of",)), load_thresholds(thresholds))
    typer.echo(json.dumps(reading.model_dump(mode="json"), indent=2))
