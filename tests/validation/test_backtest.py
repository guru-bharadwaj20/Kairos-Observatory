from datetime import date

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kairos.cli import app
from kairos.validation import (
    Detector,
    HyperbolicDetector,
    Prediction,
    as_of_view,
    decimal_year,
    from_decimal_year,
    get_detector,
    predictions_frame,
    run_backtest,
    yearly_dates,
)


def hyperbola(t_c: float = 2030.0, k: float = 1.0, start: int = 1950, end: int = 2020):
    ds = pd.date_range(f"{start}-01-01", f"{end}-01-01", freq="YS")
    t = decimal_year(ds)
    y = 1e9 * (t_c - t) ** (-k)
    return pd.DataFrame({"ds": ds, "y": y, "series_id": "toy"})


def exponential(rate: float = 0.35, start: int = 1970, end: int = 2020, noise: float = 0.0):
    ds = pd.date_range(f"{start}-01-01", f"{end}-01-01", freq="YS")
    t = decimal_year(ds)
    rng = np.random.default_rng(0)
    y = np.exp(rate * (t - start) + noise * rng.standard_normal(len(t)))
    return pd.DataFrame({"ds": ds, "y": y, "series_id": "exp"})


class Spy:
    """Records what the detector saw."""

    name = "spy"

    def __call__(self, frame, as_of):
        return Prediction(
            self.name,
            "toy",
            as_of,
            len(frame),
            None,
            None,
            None,
            None,
            False,
            0,
            {"last": frame["ds"].max()},
        )


def test_decimal_year_roundtrip():
    assert decimal_year(date(2000, 1, 1)) == 2000.0
    assert from_decimal_year(decimal_year(date(2026, 11, 13))) == date(2026, 11, 13)


def test_as_of_view_has_no_lookahead():
    frame = hyperbola()
    view = as_of_view(frame, date(1980, 6, 1))
    assert view["ds"].max() == pd.Timestamp("1980-01-01")
    lagged = as_of_view(frame, date(1980, 6, 1), publication_lag_days=365)
    assert lagged["ds"].max() == pd.Timestamp("1979-01-01")


def test_as_of_view_respects_vintages():
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2000-01-01", "2000-01-01", "2000-02-01"]),
            "y": [1.0, 2.0, 3.0],
            "series_id": "v",
            "available_from": pd.to_datetime(["2000-02-15", "2000-06-01", "2000-03-15"]),
        }
    )
    early = as_of_view(frame, date(2000, 3, 1))
    assert early["y"].tolist() == [1.0]
    late = as_of_view(frame, date(2000, 7, 1))
    assert late["y"].tolist() == [2.0, 3.0]


def test_runner_passes_only_visible_rows():
    preds = run_backtest(hyperbola(), Spy(), yearly_dates(1960, 1962))
    assert [p.detail["last"].year for p in preds] == [1960, 1961, 1962]
    assert [p.n_obs for p in preds] == [11, 12, 13]


def test_runner_rejects_many_series():
    two = pd.concat([hyperbola(), hyperbola().assign(series_id="b")])
    with pytest.raises(ValueError, match="one series"):
        run_backtest(two, Spy(), [date(2000, 1, 1)])


def test_baseline_satisfies_protocol():
    assert isinstance(HyperbolicDetector(), Detector)
    assert isinstance(get_detector("hyperbolic"), HyperbolicDetector)
    with pytest.raises(KeyError):
        get_detector("nope")


def test_baseline_recovers_known_singularity():
    pred = HyperbolicDetector()(hyperbola(t_c=2030.0, k=1.0), date(2020, 1, 1))
    assert pred.t_c == pytest.approx(2030.0, abs=0.05)
    assert pred.delta == pytest.approx(1.0, rel=1e-3)
    assert pred.alarm
    assert pred.t_c_lo <= pred.t_c <= pred.t_c_hi
    reading = pred.to_dial_reading()
    assert reading.dial == "acceleration"
    assert reading.level == pred.level >= 1


def test_baseline_quiet_on_exponential():
    pred = HyperbolicDetector()(exponential(noise=0.05), date(2020, 1, 1))
    assert not pred.alarm
    assert pred.level == 0


def test_too_few_points():
    pred = HyperbolicDetector()(hyperbola().head(3), date(1953, 1, 1))
    assert pred.t_c is None
    assert not pred.alarm


def test_parallel_matches_serial():
    dates = yearly_dates(1990, 1993)
    serial = predictions_frame(run_backtest(hyperbola(), HyperbolicDetector(), dates, n_jobs=1))
    par = predictions_frame(run_backtest(hyperbola(), HyperbolicDetector(), dates, n_jobs=2))
    pd.testing.assert_frame_equal(serial, par)


def test_cli_lists_and_rejects_unknown():
    runner = CliRunner()
    assert runner.invoke(app, ["validate", "list"]).exit_code == 0
    assert runner.invoke(app, ["validate", "run", "no-such-case"]).exit_code == 2
