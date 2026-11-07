"""H: AI safety and interpretability publications per month (``h.safety_papers``).

Counts come from OpenAlex with a **fixed, versioned query** (:data:`QUERY`). Any
change to the query must bump :data:`QUERY_VERSION`; the version is stored with the
series so that counts from different queries are never mixed silently.

OpenAlex cannot group by publication date, so the fetcher pages through the
matching works (``select=publication_date``, cursor paging) and counts them per
month locally. Works dated after the retrieval month are dropped.
"""

from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path
from typing import Final

import httpx
import pandas as pd

from kairos.data.schema import make_series, month_start
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.sources.metr import read_curated_csv
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = [
    "PUBLICATIONS",
    "QUERY",
    "QUERY_VERSION",
    "SAFETY_PAPERS",
    "count_by_month",
    "fetch_openalex_counts",
]

OPENALEX_API: Final = "https://api.openalex.org/works"
QUERY_VERSION: Final = "openalex-2026.11-v1"
QUERY: Final = {
    "title_and_abstract.search": '"AI alignment"|"AI safety"|"mechanistic interpretability"',
    "from_publication_date": "2012-01-01",
}
"""OpenAlex filter: any of the three phrases (stemmed) in title or abstract."""

SAFETY_PAPERS: Final = SeriesMeta(
    series_id="h.safety_papers",
    name="AI safety, alignment and interpretability papers per month (OpenAlex)",
    unit="papers/month",
    source="openalex",
    source_group="openalex",
    side="H",
    transform="log",
    license="CC0-1.0",
    url="https://openalex.org/",
)


def _filter_string() -> str:
    return ",".join(f"{k}:{v}" for k, v in QUERY.items())


def count_by_month(dates: pd.Series, *, until: date | None = None) -> pd.DataFrame:
    """Count publication dates per month, filling empty months with zero.

    Args:
        dates: Publication dates.
        until: Drop dates after this month (defaults to no cut).

    Returns:
        Columns ``month`` (month start) and ``count``.
    """
    months = month_start(dates)
    if until is not None:
        months = months[months <= pd.Timestamp(until).to_period("M").to_timestamp()]
    counts = months.value_counts().sort_index()
    grid = pd.date_range(counts.index.min(), counts.index.max(), freq="MS")
    counts = counts.reindex(grid, fill_value=0)
    return pd.DataFrame({"month": grid.strftime("%Y-%m-%d"), "count": counts.to_numpy()})


def fetch_openalex_counts(dest: Path, client: httpx.Client, *, today: date | None = None) -> None:
    """Page through the OpenAlex query year by year and write monthly counts to ``dest``.

    Basic paging is capped at 10,000 results per query, so each publication year is
    queried separately. ``$OPENALEX_API_KEY`` is sent when set.
    """
    today = today or date.today()
    start_year = int(QUERY["from_publication_date"][:4])
    key = os.environ.get("OPENALEX_API_KEY")
    dates: list[str] = []
    for year in range(start_year, today.year + 1):
        flt = f"{QUERY['title_and_abstract.search']}"
        params: dict[str, str | int] = {
            "filter": f"title_and_abstract.search:{flt},publication_year:{year}",
            "select": "publication_date",
            "per-page": 200,
        }
        if key:
            params["api_key"] = key
        page = 1
        while True:
            resp = client.get(OPENALEX_API, params={**params, "page": page})
            resp.raise_for_status()
            results = resp.json()["results"]
            dates += [w["publication_date"] for w in results if w.get("publication_date")]
            if len(results) < 200:
                break
            page += 1
    counts = count_by_month(pd.Series(dates), until=today)
    with dest.open("w", newline="\n", encoding="utf-8") as fh:
        fh.write(f"# OpenAlex monthly counts, query version {QUERY_VERSION}\n")
        fh.write(f"# filter={_filter_string()}\n")
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["month", "count"])
        writer.writerows(counts.itertuples(index=False))


def _build(ctx: BuildContext) -> BuildResult:
    counts = read_curated_csv(ctx.files["safety_papers_monthly.csv"])
    last_full = pd.Timestamp(ctx.vintage).to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)
    counts = counts[pd.to_datetime(counts["month"]) <= last_full]
    frame = make_series(SAFETY_PAPERS, counts["month"], counts["count"], vintage=ctx.vintage)
    extra = {"query_version": QUERY_VERSION, "query": QUERY, "api": OPENALEX_API}
    return BuildResult(datasets=[Dataset(SAFETY_PAPERS, frame, extra)])


PUBLICATIONS: Final = register(
    Source(
        name="publications",
        title="OpenAlex works matching the KAIROS safety query",
        license="CC0-1.0",
        url="https://openalex.org/",
        files=(RawFile("safety_papers_monthly.csv", fetch=fetch_openalex_counts),),
        build=_build,
    )
)
