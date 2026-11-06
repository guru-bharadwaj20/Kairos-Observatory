from datetime import date

import numpy as np
import pandas as pd
import pytest

from kairos.data.frontier import EnvelopeSpec, frontier_envelope
from kairos.data.sources.base import snapshot_path
from kairos.data.sources.epoch import frontier_compute, parse_models

REL = pd.DataFrame(
    {
        "ds": ["2020-01-10", "2020-01-20", "2020-03-05", "2020-04-01", "2021-06-01"],
        "value": [10.0, 100.0, 50.0, 1000.0, 1e6],
    }
)


def test_running_max_loglinear():
    env = frontier_envelope(REL, EnvelopeSpec(method="running_max", start=None))
    assert env["ds"].iloc[0] == pd.Timestamp("2020-01-01")
    assert env["ds"].iloc[-1] == pd.Timestamp("2021-06-01")
    assert env["record"].sum() == 3  # Jan (100), Apr (1000), Jun 2021 (1e6)
    assert env.loc[0, "y"] == pytest.approx(100.0)
    # Feb, Mar interpolated geometrically between 100 and 1000
    assert env.loc[1, "y"] == pytest.approx(100 * 10 ** (1 / 3))
    assert env["interpolated"].tolist()[:4] == [False, True, True, False]
    # Apr 2020 -> Jun 2021 is 14 months, longer than 12: interior flagged as gap
    assert env["gap"].iloc[4:-1].all()
    assert not env["gap"].iloc[:4].any()
    assert env["months_since_record"].iloc[-2] == 13
    assert np.all(np.diff(np.log(env["y"])) >= 0)


def test_step_and_top_k():
    step = frontier_envelope(
        REL, EnvelopeSpec(method="running_max", interpolation="step", start=None)
    )
    assert step.loc[1, "y"] == pytest.approx(100.0)
    top = frontier_envelope(REL, EnvelopeSpec(method="top_k", k=2, start=None))
    # Jan: geometric mean of 10 and 100
    assert top.loc[0, "y"] == pytest.approx(np.sqrt(10 * 100))


def test_no_releases():
    with pytest.raises(ValueError, match="no positive"):
        frontier_envelope(pd.DataFrame({"ds": ["2020-01-01"], "value": [0.0]}))


def test_metadata():
    md = EnvelopeSpec().as_metadata()
    assert md["method"] == "top_k"
    assert md["builder"].endswith("frontier_envelope")


def test_epoch_frontier_from_snapshot():
    models = parse_models(snapshot_path("epoch", "notable_ai_models.csv"))
    d = frontier_compute(models, date(2026, 11, 6))
    f = d.frame
    assert d.meta.series_id == "c.frontier_compute"
    assert d.extra["envelope"]["k"] == 3
    assert f["ds"].iloc[0] >= pd.Timestamp("2010-01-01")
    assert f["y"].is_monotonic_increasing
    assert f.loc[f["ds"] == "2023-03-01", "y"].iloc[0] > 1e24
