from datetime import date

import pandas as pd

from kairos.data.sources.base import BuildContext, snapshot_path
from kairos.data.sources.costs import COSTS, cost_at_threshold


def test_cost_running_min_above_threshold():
    p = pd.DataFrame(
        {
            "model": ["a", "b", "weak", "c"],
            "release_date": ["2023-01-01", "2023-06-01", "2023-07-01", "2024-01-01"],
            "mmlu": [0.9, 0.85, 0.5, 0.9],
            "input_usd_per_mtok": [10.0, 20.0, 0.01, 1.0],
            "output_usd_per_mtok": [10.0, 20.0, 0.01, 1.0],
        }
    )
    h = cost_at_threshold(p)
    assert h["model"].tolist() == ["a", "c"]
    assert h["price"].tolist() == [10.0, 1.0]


def test_snapshot_builds():
    res = COSTS.build(
        BuildContext(
            files={"model_prices.csv": snapshot_path("costs", "model_prices.csv")},
            vintage=date(2026, 11, 6),
        )
    )
    d = res.datasets[0]
    assert d.extra["direction"] == -1
    assert d.frame["y"].is_monotonic_decreasing
    assert d.frame["y"].iloc[0] == 37.5
