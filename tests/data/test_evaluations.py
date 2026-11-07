import numpy as np
import pandas as pd
import pytest

from kairos.data.sources.evaluations import evaluation_share, wilson_interval


def test_wilson_contains_point():
    lo, hi = wilson_interval(np.array([0, 1, 3]), np.array([2, 2, 3]))
    assert lo[0] == 0
    assert hi[0] > 0
    assert lo[1] < 0.5 < hi[1]
    assert hi[2] == pytest.approx(1.0)


def test_quarterly_share():
    models = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2023-01-10", "2023-02-01", "2023-05-01", "2023-06-01"]),
            "model": ["a", "b", "c", "d"],
            "frontier": [True, True, True, False],
        }
    )
    s = evaluation_share(models, {"a"})
    assert s["ds"].tolist() == [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-04-01")]
    assert s["share"].tolist() == [0.5, 0.0]
    assert s["n"].tolist() == [2, 1]
    assert (s["lo"] <= s["share"]).all()
    assert (s["hi"] >= s["share"]).all()
