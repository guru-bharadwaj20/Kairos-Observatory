from datetime import date

import pytest

from kairos.validation import run_case
from kairos.validation.population import (
    VON_FOERSTER_T_C,
    LocalDeltaDetector,
    load_population,
    sensitivity,
    thin,
)


def test_snapshot_is_valid_series_frame():
    frame = load_population()
    assert frame["ds"].min().year == 1700
    assert frame["y"].is_monotonic_increasing
    assert frame.loc[frame["ds"].dt.year == 1960, "y"].item() == pytest.approx(3.0e9, rel=0.05)


def test_thin_keeps_grid():
    years = thin(load_population(), start=1750, step=10)["ds"].dt.year
    assert set(years % 10) == {0}
    assert years.min() == 1750


def test_local_delta_quiet_after_transition():
    frame = load_population()
    pred = LocalDeltaDetector()(frame[frame["ds"].dt.year <= 1985], date(1985, 1, 1))
    assert pred.delta < 0
    assert not pred.alarm
    assert pred.t_c is None


def test_case_reproduces_von_foerster_and_delta_decline():
    result = run_case("population")
    assert result.passed, result.summary
    lo, hi = result.summary["t_c_interval_1960"]
    assert lo <= VON_FOERSTER_T_C <= hi
    assert result.summary["delta_local_1980"] < result.summary["delta_local_1960"]


def test_sensitivity_is_reported():
    table = sensitivity()
    assert {"start", "step", "t_c"} <= set(table.columns)
    assert table["t_c"].max() - table["t_c"].min() > 50  # honest: the fit is fragile
