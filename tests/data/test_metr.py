from datetime import date

import pandas as pd

from kairos.data.sources.base import BuildContext, snapshot_path
from kairos.data.sources.metr import METR, horizon_series


def test_snapshot_builds_with_intervals():
    res = METR.build(
        BuildContext(
            files={"time_horizons.csv": snapshot_path("metr", "time_horizons.csv")},
            vintage=date(2026, 11, 6),
        )
    )
    f = res.datasets[0].frame
    assert (f["y_lo"] <= f["y"]).all()
    assert (f["y_hi"] >= f["y"]).all()
    assert f["y"].iloc[-1] > 100 * f["y"].iloc[0]


def test_same_month_keeps_max():
    t = pd.DataFrame(
        {
            "model": ["a", "b"],
            "release_date": ["2024-01-03", "2024-01-20"],
            "p50_minutes": [5.0, 3.0],
            "ci_lo_minutes": [1.0, 1.0],
            "ci_hi_minutes": [9.0, 9.0],
        }
    )
    f = horizon_series(t, "2026-01-01").frame
    assert f["y"].tolist() == [5.0]
    assert f["model"].tolist() == ["a"]
