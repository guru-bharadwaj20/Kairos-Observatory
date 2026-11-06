from __future__ import annotations

import pandas as pd
import pytest

from kairos.neural import NeuralConfig, build_model, fit_model, predict, synthetic_series
from kairos.neural.events import (
    EventSpec,
    add_events,
    events_frame,
    specs_from_calendar,
    with_events,
)
from kairos.neural.regressors import design_frame


def test_spec_normalises_to_month_start():
    spec = EventSpec.of("law", ["2015-03-17", "2015-03-01", "2018-07-04"], upper_window=2)
    assert spec.dates == (pd.Timestamp("2015-03-01"), pd.Timestamp("2018-07-01"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad name"}, "alphanumeric"),
        ({"lower_window": 1}, "window"),
        ({"upper_window": -1}, "window"),
    ],
)
def test_spec_validation(kwargs, match):
    base = {"name": "ok", "dates": ["2015-01-01"]}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        EventSpec.of(base.pop("name"), base.pop("dates"), **base)


def test_events_frame_and_calendar_roundtrip():
    cal = pd.DataFrame(
        {
            "event": ["release", "release", "export_control"],
            "ds": pd.to_datetime(["2012-01-01", "2014-06-01", "2013-02-01"]),
            "upper_window": [3, 2, 0],
        }
    )
    specs = specs_from_calendar(cal)
    assert [s.name for s in specs] == ["export_control", "release"]
    assert specs[1].upper_window == 3
    ev = events_frame(specs)
    assert list(ev.columns) == ["event", "ds"]
    assert len(ev) == 3
    with pytest.raises(ValueError, match="missing"):
        specs_from_calendar(cal.drop(columns="ds"))


def test_events_wired_into_fit():
    sf = synthetic_series(60, seed=0, drivers={}, events={"law": ("2012-06-01", 0.3)})
    df = design_frame(sf, "capability", drivers=())
    spec = EventSpec.of("law", ["2012-06-01"], lower_window=-1, upper_window=2)
    cfg = NeuralConfig(epochs=5, early_stopping=False)
    model = add_events(build_model(cfg, "cpu"), [spec])
    df_ev = with_events(model, df, [spec])
    assert "law" in df_ev.columns
    assert df_ev["law"].sum() == 1
    fit_model(model, df_ev, config=cfg, seed=0)
    forecast = predict(model, df_ev)
    assert "event_law" in forecast.columns
    window = forecast.loc[forecast["event_law"] != 0, "ds"]
    assert window.min() == pd.Timestamp("2012-05-01")
    assert window.max() == pd.Timestamp("2012-08-01")


def test_with_events_noop_without_specs():
    df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=3, freq="MS"), "y": 1.0})
    assert with_events(build_model(NeuralConfig(), "cpu"), df, []) is df
