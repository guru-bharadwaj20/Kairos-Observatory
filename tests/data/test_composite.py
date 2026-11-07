import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import pytest

from kairos.data.composite import build_indices, composite_index, leave_one_out


@pytest.fixture
def comps(make_dataset):
    a = make_dataset("c.a", n=48, growth=0.05)
    b = make_dataset("c.b", n=36, start="2021-01-01", growth=0.02)
    cost = make_dataset("c.cost", n=48, growth=-0.1)
    cost = dataclasses.replace(cost, extra={"direction": -1})
    return {d.meta.series_id: d for d in (a, b, cost)}


def test_geomean_index(comps):
    res = composite_index(comps, ["c.a", "c.b", "c.cost", "c.missing"])
    idx = res.index
    assert res.components == ("c.a", "c.b", "c.cost")
    assert idx["y"].iloc[0] == pytest.approx(1.0)  # base 2020-01 has one component only
    # cost has direction -1, so its decline adds to the index
    assert (idx["growth"] > 0).all()
    assert idx["y"].is_monotonic_increasing
    assert idx["n_components"].max() == 3


def test_pca_variant(comps):
    rng = np.random.default_rng(0)
    noisy = {}
    for k, d in comps.items():
        f = d.frame.copy()
        f["y"] = f["y"] * np.exp(rng.normal(0, 0.01, len(f)))
        noisy[k] = dataclasses.replace(d, frame=f)
    res = composite_index(noisy, list(noisy), method="pca")
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert len(res.index) > 0


def test_too_few(comps):
    with pytest.raises(ValueError, match="need"):
        composite_index(comps, ["c.a"])


def test_loo(comps):
    loo = leave_one_out(comps, list(comps))
    assert loo["dropped"].tolist()[0] == "(none)"
    assert set(loo["dropped"]) == {"(none)", "c.a", "c.b", "c.cost"}


def test_build_indices(make_dataset):
    ds = {}
    for sid in ("c.frontier_compute", "c.metr_horizon", "h.safety_papers", "h.safety_funding"):
        side = sid[0].upper()
        ds[sid] = make_dataset(sid, side=side, n=60, start="2018-01-01")
    out, tables = build_indices(ds, date(2026, 1, 1))
    assert [d.meta.series_id for d in out] == ["idx.C", "idx.H"]
    assert set(tables) == {"loo_C", "loo_H"}
    assert out[1].frame["side"].eq("H").all()
    assert pd.Timestamp("2020-01-01") in set(out[0].frame["ds"])
