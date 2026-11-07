"""METR task-completion time horizons (``c.metr_horizon``).

Sparse points, one per evaluated model: the task length (human expert minutes) at
which the model succeeds 50% of the time. METR's confidence intervals are kept in
``y_lo``/``y_hi``. When several models fall in one month, the highest horizon is kept.
The input is a curated CSV (see the snapshot header for provenance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from kairos.data.schema import make_series, month_start
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = ["METR", "METR_HORIZON", "horizon_series", "read_curated_csv"]

METR_URL: Final = "https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/"

METR_HORIZON: Final = SeriesMeta(
    series_id="c.metr_horizon",
    name="METR 50% task-completion time horizon",
    unit="minutes",
    source="metr",
    source_group="metr",
    side="C",
    transform="log",
    license="METR publication terms",
    url=METR_URL,
)


def read_curated_csv(path: Path) -> pd.DataFrame:
    """Read a curated CSV whose ``#`` lines are provenance comments."""
    return pd.read_csv(path, comment="#", skipinitialspace=True)


def horizon_series(table: pd.DataFrame, vintage: object) -> Dataset:
    """Turn the curated horizon table into the ``c.metr_horizon`` dataset.

    Args:
        table: Columns ``model``, ``release_date``, ``p50_minutes``,
            ``ci_lo_minutes``, ``ci_hi_minutes``.
        vintage: Retrieval/curation date.
    """
    df = table.copy()
    df["ds"] = month_start(df["release_date"]).to_numpy()
    df = df.sort_values(["ds", "p50_minutes"]).groupby("ds", as_index=False).last()
    frame = make_series(
        METR_HORIZON,
        df["ds"],
        df["p50_minutes"],
        vintage=str(vintage),
        y_lo=df["ci_lo_minutes"],
        y_hi=df["ci_hi_minutes"],
        extra={"model": df["model"].astype(str).tolist()},
    )
    return Dataset(METR_HORIZON, frame, {"metric": "p50 time horizon", "curated": True})


def _build(ctx: BuildContext) -> BuildResult:
    table = read_curated_csv(ctx.files["time_horizons.csv"])
    return BuildResult(datasets=[horizon_series(table, ctx.vintage)])


METR: Final = register(
    Source(
        name="metr",
        title="METR, task-completion time horizons",
        license="METR publication terms",
        url=METR_URL,
        files=(RawFile("time_horizons.csv"),),
        build=_build,
    )
)
