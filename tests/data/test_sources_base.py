from datetime import date

import httpx
import pytest

from kairos.data.sources import base
from kairos.data.sources.epoch import EPOCH, MODELS_TABLE, parse_models


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_offline_uses_snapshot(tmp_path):
    fetched = base.fetch_source(EPOCH, tmp_path, offline=True)
    assert fetched.origin == {"notable_ai_models.csv": "snapshot"}
    assert fetched.vintage == base.snapshot_vintage("epoch")
    assert fetched.files["notable_ai_models.csv"].read_bytes() == (
        base.snapshot_path("epoch", "notable_ai_models.csv").read_bytes()
    )


def test_live_download(tmp_path):
    body = b"Model,Publication date,Training compute (FLOP)\nX,2020-05-01,1e20\n"
    client = _client(lambda req: httpx.Response(200, content=body))
    fetched = base.fetch_source(EPOCH, tmp_path, client=client, today=date(2030, 1, 1))
    assert fetched.origin["notable_ai_models.csv"] == "live"
    assert fetched.vintage == date(2030, 1, 1)
    models = parse_models(fetched.files["notable_ai_models.csv"])
    assert models["training_compute_flop"].tolist() == [1e20]
    assert models["organization"].tolist() == [""]


def test_failed_download_falls_back(tmp_path):
    client = _client(lambda req: httpx.Response(503))
    fetched = base.fetch_source(EPOCH, tmp_path, client=client)
    assert fetched.origin["notable_ai_models.csv"] == "snapshot"


def test_registry_order_and_cycles(monkeypatch):
    sources = base.registered_sources()
    assert "epoch" in sources
    names = list(sources)
    for name, src in sources.items():
        for dep in src.requires:
            assert names.index(dep) < names.index(name)
    assert base.get_source("epoch") is EPOCH
    with pytest.raises(KeyError):
        base.get_source("nope")


def test_epoch_snapshot_parses():
    fetched_path = base.snapshot_path("epoch", "notable_ai_models.csv")
    models = parse_models(fetched_path)
    assert len(models) > 300
    assert models["ds"].is_monotonic_increasing
    assert models["frontier"].any()
    gpt3 = models[models["model"] == "GPT-3 175B (davinci)"]
    assert not gpt3.empty
    assert 1e23 < gpt3["training_compute_flop"].iloc[0] < 1e24
    result = EPOCH.build(
        base.BuildContext(files={"notable_ai_models.csv": fetched_path}, vintage=date(2026, 1, 1))
    )
    assert MODELS_TABLE in result.tables
