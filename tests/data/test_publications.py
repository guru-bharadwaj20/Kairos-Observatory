from datetime import date

import httpx
import pandas as pd

from kairos.data.sources.base import BuildContext, fetch_source
from kairos.data.sources.publications import (
    PUBLICATIONS,
    QUERY_VERSION,
    count_by_month,
    fetch_openalex_counts,
)


def test_count_by_month_fills_zeros():
    c = count_by_month(
        pd.Series(["2020-01-05", "2020-03-02", "2020-03-09", "2021-01-01"]),
        until=date(2020, 12, 31),
    )
    assert c["month"].tolist() == ["2020-01-01", "2020-02-01", "2020-03-01"]
    assert c["count"].tolist() == [1, 0, 2]


def test_fetch_pages_per_year(tmp_path):
    seen = []

    def handler(req):
        flt = req.url.params["filter"]
        page = int(req.url.params["page"])
        seen.append((flt.rsplit(":", 1)[1], page))
        n = 200 if page == 1 and flt.endswith("2025") else 3
        return httpx.Response(200, json={"results": [{"publication_date": "2025-02-10"}] * n})

    dest = tmp_path / "c.csv"
    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetch_openalex_counts(dest, client, today=date(2026, 3, 1))
    assert ("2025", 2) in seen
    assert ("2024", 2) not in seen
    text = dest.read_text()
    assert QUERY_VERSION in text
    assert "2025-02-01,245" in text  # 15 years x 3 + 200 extra, all dated Feb 2025


def test_snapshot_build_drops_partial_month(tmp_path):
    f = fetch_source(PUBLICATIONS, tmp_path, offline=True)
    (d,) = PUBLICATIONS.build(BuildContext(files=f.files, vintage=date(2026, 11, 6))).datasets
    assert d.frame["ds"].max() == pd.Timestamp("2026-10-01")
    assert d.extra["query_version"] == QUERY_VERSION
    assert d.meta.side == "H"
