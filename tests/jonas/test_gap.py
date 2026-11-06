from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.jonas import jonas_gap, side_components, split_sides


def test_gap_zero_in_base_year_and_slope(frame_factory) -> None:
    frame = frame_factory()
    gap = jonas_gap(frame, base_year=2016)
    base = gap.loc[gap["ds"].dt.year == 2016, "gap"]
    assert base.mean() == pytest.approx(0.0, abs=1e-12)
    # C grows at a mean 0.8/yr, H at 0.25/yr, so G rises at 0.55/yr.
    slope = np.polyfit(np.arange(len(gap)) / 12.0, gap["gap"], 1)[0]
    assert slope == pytest.approx(0.55, rel=1e-9)


def test_level_is_unit_free(frame_factory) -> None:
    frame = frame_factory()
    scaled = frame.copy()
    scaled.loc[scaled["series_id"] == "h_papers", "y"] *= 1e6
    a = jonas_gap(frame, 2016)["gap"].to_numpy()
    b = jonas_gap(scaled, 2016)["gap"].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_weights_and_missing_months(frame_factory) -> None:
    frame = frame_factory()
    drop = (frame["series_id"] == "c_bench") & (frame["ds"] == pd.Timestamp("2020-03-01"))
    gap = jonas_gap(frame.loc[~drop], 2016, weights={"c_bench": 0.0})
    full = jonas_gap(frame.loc[frame["series_id"] != "c_bench"], 2016)
    np.testing.assert_allclose(gap["gap"], full["gap"], atol=1e-12)
    assert set(gap.columns) == {"ds", "ln_c", "ln_h", "gap", "n_c", "n_h"}


def test_errors(frame_factory) -> None:
    frame = frame_factory()
    with pytest.raises(ValueError, match="base year"):
        jonas_gap(frame, 1990)
    with pytest.raises(ValueError, match="no H series"):
        split_sides(frame.loc[frame["source_group"] == "C"])
    with pytest.raises(ValueError, match="source_group"):
        split_sides(frame.drop(columns="source_group"))
    assert side_components(frame) == {
        "C": ["c_bench", "c_compute"],
        "H": ["h_funding", "h_papers"],
    }
