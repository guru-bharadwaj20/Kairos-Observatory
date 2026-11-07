from datetime import date

import numpy as np
import pandas as pd
import pytest

from kairos.data.sources.base import BuildContext, snapshot_path
from kairos.data.sources.benchmarks import BENCHMARKS, sota_history
from kairos.data.transforms import annual_to_monthly, inv_logit, logit, transform_values


def test_sota_keeps_records_only():
    s = pd.DataFrame(
        {
            "release_date": ["2020-01-01", "2020-03-01", "2020-05-01", "2020-05-20"],
            "score": [0.5, 0.4, 0.7, 0.6],
            "model": list("abcd"),
        }
    )
    assert sota_history(s)["model"].tolist() == ["a", "c"]


def test_logit_roundtrip_and_saturation():
    p = np.array([0.5, 0.9, 0.99])
    np.testing.assert_allclose(inv_logit(logit(p)), p)
    # 0.9 -> 0.99 is as large a step on the logit scale as 0.5 -> 0.9 roughly
    steps = np.diff(logit(p))
    assert steps[1] > 0.9 * steps[0]
    assert np.isfinite(logit([0.0, 1.0])).all()


def test_transform_values():
    assert transform_values([np.e], "log")[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown"):
        transform_values([1.0], "sqrt")


def test_annual_to_monthly():
    m = annual_to_monthly([2020, 2021], [1.0, 4.0])
    assert m["ds"].iloc[0] == pd.Timestamp("2020-07-01")
    assert len(m) == 13
    assert m["y"].iloc[6] == pytest.approx(2.0)
    assert m["interpolated"].sum() == 11


def test_snapshot_builds():
    res = BENCHMARKS.build(
        BuildContext(
            files={"benchmark_scores.csv": snapshot_path("benchmarks", "benchmark_scores.csv")},
            vintage=date(2026, 11, 6),
        )
    )
    ids = {d.meta.series_id for d in res.datasets}
    assert ids == {"c.bench.mmlu", "c.bench.gpqa_diamond", "c.bench.swe_bench_verified"}
    for d in res.datasets:
        assert d.meta.transform == "logit"
        assert d.frame["y"].is_monotonic_increasing
        assert d.frame["y"].between(0, 1).all()
