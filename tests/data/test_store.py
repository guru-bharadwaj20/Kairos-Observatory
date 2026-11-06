import pandas as pd
import pytest

from kairos.data import store


def test_roundtrip(root, make_dataset):
    d = make_dataset()
    path = store.write_dataset(d, root)
    assert path == root / "processed" / "series" / "c.test.parquet"
    back = store.read_series("c.test", root)
    pd.testing.assert_frame_equal(back, d.frame)
    assert store.read_meta("c.test", root) == d.meta
    assert store.read_extra("c.test", root) == {"note": "synthetic"}


def test_mismatched_id_rejected(root, make_dataset):
    d = make_dataset()
    bad = store.Dataset(make_dataset("c.other").meta, d.frame)
    with pytest.raises(ValueError, match="series_id"):
        store.write_dataset(bad, root)


def test_missing_series(root, make_dataset):
    with pytest.raises(FileNotFoundError):
        store.read_series("nope", root)


def test_catalog(root, make_dataset):
    store.write_dataset(make_dataset("c.a"), root)
    store.write_dataset(make_dataset("h.b", side="H", n=12), root)
    assert store.list_series(root) == ["c.a", "h.b"]
    assert set(store.read_all(root, side="H")) == {"h.b"}
    con = store.connect_catalog(root)
    rows = con.execute(
        "SELECT m.side, count(*) FROM series s JOIN series_meta m USING (series_id) "
        "GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert rows == [("C", 24), ("H", 12)]


def test_catalog_empty_and_events(root, make_dataset):
    from kairos.data.events import validate_events

    con = store.connect_catalog(root)
    assert con.execute("SELECT count(*) FROM duckdb_views() WHERE NOT internal").fetchone() == (0,)
    ev = validate_events(
        pd.DataFrame(
            {
                "event": ["a"],
                "ds": pd.to_datetime(["2020-01-02"]).astype("datetime64[ns]"),
                "type": ["release"],
                "lower_window": [0],
                "upper_window": [0],
                "source_url": ["https://x.org"],
                "scope": ["ai"],
                "description": [""],
            }
        )
    )
    store.write_events(ev, root)
    assert store.connect_catalog(root).execute("SELECT event FROM events").fetchall() == [("a",)]


def test_tables(root, make_dataset):
    df = pd.DataFrame({"a": [1, 2]})
    store.write_table(df, "t", root)
    pd.testing.assert_frame_equal(store.read_table("t", root), df)


def test_default_root_from_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_DATA_DIR", str(tmp_path / "d"))
    assert store.data_paths().root == tmp_path / "d"
