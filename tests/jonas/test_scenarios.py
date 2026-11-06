from __future__ import annotations

import json

import numpy as np
import pytest

from kairos.jonas import SCENARIO_COLUMNS, closing_scenarios, gap_velocity, to_pct


def test_exact_scenarios(frame_factory) -> None:
    sc = closing_scenarios(frame_factory(), 2016)
    assert sc.g_c == pytest.approx(0.8)
    assert sc.g_h == pytest.approx(0.25)
    assert sc.required_rate == pytest.approx(0.8)
    assert sc.shortfall == pytest.approx(0.55)
    assert sc.required_pct == pytest.approx(100 * (np.exp(0.8) - 1))
    comps = sc.components.set_index("series_id")
    assert tuple(sc.components.columns) == SCENARIO_COLUMNS
    # papers alone: (0.8 - 0.5*0.2)/0.5 = 1.4; funding alone: (0.8 - 0.5*0.3)/0.5 = 1.3
    assert comps.loc["h_papers", "required_rate"] == pytest.approx(1.4)
    assert comps.loc["h_funding", "required_rate"] == pytest.approx(1.3)
    assert comps.loc["h_papers", "multiplier"] == pytest.approx(1.4 / 0.3)


def test_applying_required_rate_closes_gap(frame_factory) -> None:
    sc = closing_scenarios(frame_factory(noise=0.02, seed=3), 2016)
    closed = frame_factory({"c": sc.g_c}, {"h": sc.required_rate})
    vel = gap_velocity(closed, 2016, n_boot=0)
    assert vel["dgdt"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_intervals_and_serialisation(frame_factory) -> None:
    sc = closing_scenarios(frame_factory(noise=0.03, seed=5), 2016, level=0.9)
    assert sc.required_lo < sc.required_rate < sc.required_hi
    assert sc.shortfall_lo < sc.shortfall < sc.shortfall_hi
    c = sc.components
    assert (c["required_lo"] < c["required_rate"]).all()
    assert (c["required_rate"] < c["required_hi"]).all()
    json.dumps(sc.as_dict())
    wider = closing_scenarios(frame_factory(noise=0.03, seed=5), 2016, level=0.99)
    assert wider.required_hi > sc.required_hi
    with pytest.raises(ValueError, match="level"):
        closing_scenarios(frame_factory(), 2016, level=0)


def test_to_pct() -> None:
    assert to_pct(0.0) == 0.0
    assert to_pct(np.log(2)) == pytest.approx(100.0)
