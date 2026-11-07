"""Historical validation cases (side ``validation``).

Growth cases (K-2.18), source ``validation_growth``:

* ``val.population``: world population, annual from 1700 (OWID, built on HYDE and
  UN WPP). Earlier points, back to 10,000 BCE, cannot be represented as
  ``datetime64[ns]`` and are kept in the ``population_history`` table
  (``year``, ``population``) for von Foerster-style fits (K-9.02).
* ``val.transistors``: transistors per microprocessor (OWID, after Rupp), annual.

Annual values are dated at January of their year.

Bubble and epidemic cases (K-2.19), source ``validation_bubbles``:

* ``val.nasdaq``: NASDAQ Composite, monthly mean of daily closes, 1995-2001 (FRED).
* ``val.bitcoin``: Bitcoin in USD (Coinbase), monthly mean, 2015-2018 (FRED).
* ``val.covid_cases``: world new COVID-19 cases per month, 2020-2022 (OWID), with
  lockdown and vaccine events (scope ``covid``) in the event calendar.
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
    "VALIDATION_BUBBLES",
    "VALIDATION_GROWTH",
    "monthly_mean",
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


FRED: Final = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
OWID_COVID: Final = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"

NASDAQ: Final = _val("val.nasdaq", "NASDAQ Composite (monthly mean)", "index", "fred_nasdaq",
                     "https://fred.stlouisfed.org/series/NASDAQCOM",
                     "FRED terms; index data (c) Nasdaq")  # fmt: skip
BITCOIN: Final = _val("val.bitcoin", "Bitcoin price, Coinbase (monthly mean)", "USD",
                      "fred_bitcoin", "https://fred.stlouisfed.org/series/CBBTCUSD",
                      "FRED terms; data (c) Coinbase")  # fmt: skip
COVID_CASES: Final = _val("val.covid_cases", "World new COVID-19 cases per month", "cases/month",
                          "owid_covid", "https://ourworldindata.org/covid-cases")  # fmt: skip

WINDOWS: Final = {
    "val.nasdaq": ("1995-01-01", "2001-12-01"),
    "val.bitcoin": ("2015-01-01", "2018-12-01"),
    "val.covid_cases": ("2020-01-01", "2022-12-01"),
}


def monthly_mean(dates: pd.Series, values: pd.Series, *, how: str = "mean") -> pd.DataFrame:
    """Aggregate a daily series to months (``mean`` or ``sum``), dropping missing days.

    Returns:
        Columns ``ds`` (month start) and ``y``.
    """
    v = pd.to_numeric(values, errors="coerce")
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "v": v}).dropna()
    g = df.groupby(df["ds"].dt.to_period("M"))["v"].agg(how)
    return pd.DataFrame({"ds": g.index.to_timestamp().astype("datetime64[ns]"), "y": g.to_numpy()})


def _windowed(meta: SeriesMeta, m: pd.DataFrame, vintage: date) -> Dataset:
    lo, hi = WINDOWS[meta.series_id]
    m = m[(m["ds"] >= lo) & (m["ds"] <= hi) & (m["y"] > 0)]
    frame = make_series(meta, m["ds"], m["y"], vintage=vintage)
    return Dataset(meta, frame, {"window": [lo, hi], "aggregation": "monthly"})


def _build_bubbles(ctx: BuildContext) -> BuildResult:
    from kairos.data.events import read_event_csv

    nq = pd.read_csv(ctx.files["nasdaq.csv"])
    btc = pd.read_csv(ctx.files["bitcoin.csv"])
    cov = pd.read_csv(ctx.files["covid_cases.csv"], usecols=["country", "date", "new_cases"])
    cov = cov[cov["country"] == "World"]
    return BuildResult(
        datasets=[
            _windowed(NASDAQ, monthly_mean(nq.iloc[:, 0], nq["NASDAQCOM"]), ctx.vintage),
            _windowed(BITCOIN, monthly_mean(btc.iloc[:, 0], btc["CBBTCUSD"]), ctx.vintage),
            _windowed(
                COVID_CASES, monthly_mean(cov["date"], cov["new_cases"], how="sum"), ctx.vintage
            ),
        ],
        events=read_event_csv(ctx.files["lockdowns.csv"]),
    )


VALIDATION_BUBBLES: Final = register(
    Source(
        name="validation_bubbles",
        title="Validation: Nasdaq 1995-2001, Bitcoin 2015-2018, COVID-19 cases",
        license="FRED terms (Nasdaq, Coinbase); CC-BY-4.0 (OWID)",
        url="https://fred.stlouisfed.org/",
        files=(
            RawFile("nasdaq.csv", f"{FRED}NASDAQCOM"),
            RawFile("bitcoin.csv", f"{FRED}CBBTCUSD"),
            RawFile("covid_cases.csv", OWID_COVID),
            RawFile("lockdowns.csv"),
        ),
        build=_build_bubbles,
    )
)
