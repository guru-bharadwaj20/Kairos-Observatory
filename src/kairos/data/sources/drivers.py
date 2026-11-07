"""Exogenous drivers for the recursion dial ``κ`` (side ``exo``).

=================== ========================================================= =======
series_id           definition                                                unit
=================== ========================================================= =======
``exo.compute``     training compute of notable models released in the        FLOP
                    trailing 12 months (Epoch AI)
``exo.investment``  world private investment in AI, annual (OWID / Quid),     USD/yr
                    log-linearly interpolated to months
``exo.researchers`` NeurIPS main-track submissions, annual, interpolated     papers/yr
                    (proxy for the active research workforce)
=================== ========================================================= =======

:func:`exogenous_regressors` returns the wide, **logged** frame NeuralProphet wants:
``ds``, ``log_compute``, ``log_investment``, ``log_researchers``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from kairos.data.schema import make_series, month_start
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.sources.epoch import EPOCH_LICENSE, EPOCH_URL, MODELS_TABLE
from kairos.data.sources.metr import read_curated_csv
from kairos.data.store import Dataset
from kairos.data.transforms import annual_to_monthly
from kairos.types import SeriesMeta

__all__ = [
    "DRIVERS",
    "EXO_COMPUTE",
    "EXO_INVESTMENT",
    "EXO_RESEARCHERS",
    "REGRESSOR_NAMES",
    "exogenous_regressors",
    "trailing_compute",
]

OWID_INVESTMENT: Final = (
    "https://ourworldindata.org/grapher/private-investment-in-artificial-intelligence.csv"
    "?useColumnShortNames=true"
)


def _meta(sid: str, name: str, unit: str, source: str, lic: str, url: str) -> SeriesMeta:
    return SeriesMeta(sid, name, unit, source, source, "exo", "log", lic, url)


EXO_COMPUTE: Final = _meta(
    "exo.compute", "Training compute of notable models, trailing 12 months", "FLOP",
    "epoch", EPOCH_LICENSE, EPOCH_URL,
)  # fmt: skip
EXO_INVESTMENT: Final = _meta(
    "exo.investment", "World private investment in AI", "USD/yr", "owid", "CC-BY-4.0",
    "https://ourworldindata.org/grapher/private-investment-in-artificial-intelligence",
)  # fmt: skip
EXO_RESEARCHERS: Final = _meta(
    "exo.researchers", "NeurIPS main-track submissions (researcher proxy)", "papers/yr",
    "neurips", "Facts from conference statistics", "https://papers.nips.cc/",
)  # fmt: skip

REGRESSOR_NAMES: Final = {
    "exo.compute": "log_compute",
    "exo.investment": "log_investment",
    "exo.researchers": "log_researchers",
}


def trailing_compute(
    models: pd.DataFrame, *, months: int = 12, start: str = "2010-01-01"
) -> pd.DataFrame:
    """Sum training compute over a trailing window of release months.

    Args:
        models: The ``epoch_models`` table.
        months: Window length.
        start: First month of the output.

    Returns:
        Monthly ``ds``, ``y`` (months with a positive sum) up to the last release month.
    """
    m = models.dropna(subset=["training_compute_flop"])
    per_month = m.groupby(month_start(m["ds"]).to_numpy())["training_compute_flop"].sum()
    grid = pd.date_range(start, per_month.index.max(), freq="MS")
    rolled = per_month.reindex(grid, fill_value=0.0).rolling(months, min_periods=1).sum()
    out = pd.DataFrame({"ds": grid.astype("datetime64[ns]"), "y": rolled.to_numpy()})
    return out[out["y"] > 0].reset_index(drop=True)


def _annual(meta: SeriesMeta, path: Path, vintage: date, value: str, month: int) -> Dataset:
    raw = read_curated_csv(path)
    if "entity" in raw:
        raw = raw[raw["entity"] == "World"]
    m = annual_to_monthly(raw["year"].astype("int64"), raw[value], month=month)
    lo = hi = None
    if {"lo", "hi"} <= set(raw.columns):
        ratio_lo = annual_to_monthly(
            raw["year"].astype("int64"), raw["lo"] / raw[value], month=month
        )
        ratio_hi = annual_to_monthly(
            raw["year"].astype("int64"), raw["hi"] / raw[value], month=month
        )
        lo, hi = m["y"] * ratio_lo["y"], m["y"] * ratio_hi["y"]
    frame = make_series(
        meta, m["ds"], m["y"], vintage=vintage, y_lo=lo, y_hi=hi,
        extra={"interpolated": m["interpolated"].tolist()},
    )  # fmt: skip
    return Dataset(meta, frame, {"interpolation": "log-linear between annual points"})


def _build(ctx: BuildContext) -> BuildResult:
    tc = trailing_compute(ctx.tables[MODELS_TABLE])
    compute = Dataset(
        EXO_COMPUTE,
        make_series(EXO_COMPUTE, tc["ds"], tc["y"], vintage=ctx.vintage),
        {"window_months": 12},
    )
    return BuildResult(
        datasets=[
            compute,
            _annual(EXO_INVESTMENT, ctx.files["ai_private_investment.csv"], ctx.vintage,
                    "private_investment", 7),
            _annual(EXO_RESEARCHERS, ctx.files["neurips_submissions.csv"], ctx.vintage,
                    "value", 5),
        ]
    )  # fmt: skip


def exogenous_regressors(datasets: Mapping[str, Dataset]) -> pd.DataFrame:
    """Join the driver series into the logged wide frame used as lagged regressors.

    Args:
        datasets: Must contain ``exo.compute``, ``exo.investment`` and ``exo.researchers``.

    Returns:
        ``ds`` plus one ``log_*`` column per driver, on months where all exist.
    """
    wide: pd.DataFrame | None = None
    for sid, col in REGRESSOR_NAMES.items():
        f = datasets[sid].frame
        part = pd.DataFrame({"ds": f["ds"], col: np.log(f["y"].to_numpy())})
        wide = part if wide is None else wide.merge(part, on="ds", how="inner")
    assert wide is not None
    return wide.sort_values("ds").reset_index(drop=True)


DRIVERS: Final = register(
    Source(
        name="drivers",
        title="Exogenous drivers: compute, investment, researchers",
        license="CC-BY-4.0 (Epoch AI, OWID); facts (NeurIPS)",
        url="https://ourworldindata.org/artificial-intelligence",
        files=(
            RawFile("ai_private_investment.csv", OWID_INVESTMENT),
            RawFile("neurips_submissions.csv"),
        ),
        build=_build,
        requires=("epoch",),
    )
)
