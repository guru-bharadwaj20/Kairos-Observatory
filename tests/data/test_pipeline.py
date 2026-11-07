import httpx
import pytest
from typer.testing import CliRunner

from kairos.data import store
from kairos.data.manifest import verify_manifest
from kairos.data.pipeline import refresh, resolve_sources


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    return root, refresh(root, offline=True)


def test_offline_refresh_builds_everything(built):
    root, res = built
    assert {"idx.C", "idx.H", "c.frontier_compute", "h.safety_papers", "val.covid_cases"} <= set(
        res.series
    )
    assert set(res.origins.values()) == {"snapshot"}
    assert res.n_events > 30
    assert verify_manifest(root) == []
    con = store.connect_catalog(root)
    sides = dict(con.execute("SELECT side, count(*) FROM series_meta GROUP BY 1").fetchall())
    assert set(sides) == {"C", "H", "exo", "validation"}
    assert store.read_table_meta("epoch_models", root)["license"] == "CC-BY-4.0"


def test_refresh_is_reproducible(built, tmp_path):
    _, first = built
    assert refresh(tmp_path, offline=True).digest == first.digest


def test_subset_and_dependencies(tmp_path):
    names = [s.name for s in resolve_sources(["evaluations"])]
    assert names == ["epoch", "evaluations"]
    res = refresh(tmp_path, offline=True, sources=["metr"])
    assert res.series == ["c.metr_horizon"]
    with pytest.raises(KeyError, match="unknown"):
        resolve_sources(["nope"])


def test_live_failure_falls_back(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    res = refresh(tmp_path, sources=["epoch"], client=client)
    assert res.origins == {"epoch": "snapshot"}


def test_cli(built, tmp_path):
    from kairos.cli import app

    root, _ = built
    runner = CliRunner()
    out = runner.invoke(app, ["data", "verify", "--data-dir", str(root)])
    assert out.exit_code == 0, out.output
    out = runner.invoke(app, ["data", "list", "--data-dir", str(root)])
    assert out.exit_code == 0
    assert "idx.C" in out.output
    out = runner.invoke(app, ["data", "refresh", "--offline", "-s", "metr", "--data-dir",
                              str(tmp_path)])  # fmt: skip
    assert out.exit_code == 0, out.output
    (tmp_path / "processed/series/c.metr_horizon.parquet").write_bytes(b"x")
    out = runner.invoke(app, ["data", "verify", "--data-dir", str(tmp_path)])
    assert out.exit_code == 1
