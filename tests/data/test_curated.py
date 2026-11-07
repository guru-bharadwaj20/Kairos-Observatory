from datetime import date

import pytest

from kairos.data.sources.base import BuildContext, fetch_source, registered_sources


@pytest.mark.parametrize(
    "name",
    sorted(n for n, s in registered_sources().items() if all(f.url is None for f in s.files)),
)
def test_curated_sources_build_offline(name, tmp_path):
    src = registered_sources()[name]
    upstream = {}
    tables = {}
    for dep in src.requires:
        d = registered_sources()[dep]
        f = fetch_source(d, tmp_path, offline=True)
        r = d.build(BuildContext(files=f.files, vintage=f.vintage))
        upstream.update({x.meta.series_id: x for x in r.datasets})
        tables.update(r.tables)
    fetched = fetch_source(src, tmp_path, offline=True)
    res = src.build(
        BuildContext(
            files=fetched.files, vintage=date(2026, 11, 6), upstream=upstream, tables=tables
        )
    )
    assert res.datasets or res.events is not None or res.tables
    for d in res.datasets:
        assert len(d.frame) > 0
        assert d.meta.license


def test_research_share_documented(tmp_path):
    from kairos.data.sources.research_share import RESEARCH_SHARE

    f = fetch_source(RESEARCH_SHARE, tmp_path, offline=True)
    (d,) = RESEARCH_SHARE.build(BuildContext(files=f.files, vintage=f.vintage)).datasets
    assert d.extra["proxy"] is True
    assert len(d.extra["caveats"]) >= 3
    assert d.frame["y_lo"].notna().all()
