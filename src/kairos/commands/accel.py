"""``kairos accel``: acceleration-exponent tools from the command line.

* ``kairos accel delta FILE`` estimates ``δ`` and ``t_c`` with block-bootstrap
  intervals for every series in a ``SeriesFrame`` file (CSV or Parquet).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from kairos.accel.bootstrap import BootstrapResult, block_bootstrap
from kairos.accel.growth import frame_arrays

__all__ = ["app", "read_series_frame"]

app = typer.Typer(help="Acceleration dial: δ, t_c and LPPLS.", no_args_is_help=True)


@app.callback()
def _group() -> None:
    """Acceleration dial: δ, t_c and LPPLS."""


def read_series_frame(path: Path) -> pd.DataFrame:
    """Read a ``SeriesFrame`` from CSV or Parquet, parsing ``ds`` as dates."""
    parquet = path.suffix.lower() in {".parquet", ".pq"}
    df = pd.read_parquet(path) if parquet else pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)
    return df


def _bootstrap(df: pd.DataFrame, sid: str, n_boot: int, device: str | None) -> BootstrapResult:
    t, log_y = frame_arrays(df, sid)
    if device is not None:
        from kairos.accel.bootstrap_gpu import block_bootstrap_torch

        return block_bootstrap_torch(t, log_y, n_boot=n_boot, device=device)
    return block_bootstrap(t, log_y, n_boot=n_boot)


def _fmt(x: float) -> str:
    return f"{x:.3f}" if math.isfinite(x) else "∞" if x > 0 else "—"


@app.command("delta")
def delta_cmd(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="SeriesFrame file.")],
    series_id: Annotated[
        list[str] | None, typer.Option("--series-id", "-s", help="Series to fit.")
    ] = None,
    n_boot: Annotated[int, typer.Option(help="Bootstrap draws.")] = 1000,
    level: Annotated[float, typer.Option(help="Interval level.")] = 0.9,
    device: Annotated[
        str | None, typer.Option(help="Run the torch bootstrap on this device.")
    ] = None,
) -> None:
    """Estimate δ and t_c with moving-block-bootstrap intervals."""
    df = read_series_frame(path)
    ids = series_id or sorted(df["series_id"].astype(str).unique())
    table = Table(title=f"Acceleration exponent ({level:.0%} intervals)")
    for col in ("series_id", "δ", "δ lo", "δ hi", "p(δ≤0)", "t_c lo", "t_c hi"):
        table.add_column(col)
    for sid in ids:
        res = _bootstrap(df, sid, n_boot, device)
        dlo, dhi = res.interval(level)
        tlo, thi = res.interval(level, "t_c")
        table.add_row(
            sid,
            _fmt(res.point.delta),
            _fmt(dlo),
            _fmt(dhi),
            f"{res.p_delta_positive:.4f}",
            _fmt(tlo),
            _fmt(thi),
        )
    Console().print(table)
