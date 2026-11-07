"""Historical validation cases (side ``validation``).

Growth cases (K-2.18), source ``validation_growth``:

* ``val.population``: world population, annual from 1700 (OWID, built on HYDE and
  UN WPP). Earlier points, back to 10,000 BCE, cannot be represented as
  ``datetime64[ns]`` and are kept in the ``population_history`` table
  (``year``, ``population``) for von Foerster-style fits (K-9.02).
* ``val.transistors``: transistors per microprocessor (OWID, after Rupp), annual.

Annual values are dated at January of their year.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd

from kairos.data.schema import make_series
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = [
    "POPULATION",
    "POPULATION_TABLE",
    "TRANSISTORS",
    "VALIDATION_GROWTH",
    "read_owid_world",
]

OWID: Final = "https://ourworldindata.org/grapher/"
POPULATION_TABLE: Final = "population_history"
FIRST_DATED_YEAR: Final = 1700


def _val(
    sid: str, name: str, unit: str, group: str, url: str, lic: str = "CC-BY-4.0"
) -> SeriesMeta:
    return SeriesMeta(sid, name, unit, group, group, "validation", "log", lic, url)


POPULATION: Final = _val("val.population", "World population", "people", "owid_population",
                         f"{OWID}population")  # fmt: skip
TRANSISTORS: Final = _val("val.transistors", "Transistors per microprocessor", "transistors",
                          "owid_transistors", f"{OWID}transistors-per-microprocessor")  # fmt: skip


def read_owid_world(path: Path, value: str) -> pd.DataFrame:
    """Read an OWID grapher CSV and keep the ``World`` rows as ``year``, ``value``."""
    raw = pd.read_csv(path)
    world = raw[raw["entity"] == "World"]
    return pd.DataFrame(
        {"year": world["year"].astype("int64"), "value": world[value].astype(float)}
    ).reset_index(drop=True)


def _annual(meta: SeriesMeta, table: pd.DataFrame, vintage: date) -> Dataset:
    t = table[(table["year"] >= FIRST_DATED_YEAR) & (table["year"] <= vintage.year)]
    ds = pd.to_datetime(t["year"].astype(str) + "-01-01")
    return Dataset(meta, make_series(meta, ds, t["value"], vintage=vintage),
                   {"frequency": "annual (January)"})  # fmt: skip


def _build_growth(ctx: BuildContext) -> BuildResult:
    pop = read_owid_world(ctx.files["population.csv"], "population_historical")
    tr = read_owid_world(ctx.files["transistors.csv"], "transistors")
    return BuildResult(
        datasets=[_annual(POPULATION, pop, ctx.vintage), _annual(TRANSISTORS, tr, ctx.vintage)],
        tables={POPULATION_TABLE: pop.rename(columns={"value": "population"})},
    )


VALIDATION_GROWTH: Final = register(
    Source(
        name="validation_growth",
        title="Validation: world population and Moore's law (Our World in Data)",
        license="CC-BY-4.0",
        url="https://ourworldindata.org/",
        files=(
            RawFile("population.csv", f"{OWID}population.csv?useColumnShortNames=true"),
            RawFile(
                "transistors.csv",
                f"{OWID}transistors-per-microprocessor.csv?useColumnShortNames=true",
            ),
        ),
        build=_build_growth,
    )
)
