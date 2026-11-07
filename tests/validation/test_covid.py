from datetime import date

import numpy as np

from kairos.validation import run_case
from kairos.validation.covid import event_design, fit_event_model, load_italy


def test_snapshot_loads():
    data = load_italy()
    assert data["y"].max() > 30_000
    assert data["ds"].is_monotonic_increasing


def test_hinge_recovers_slopes():
    ds = load_italy()["ds"]
    x = event_design(ds)
    y = np.exp(x @ np.array([5.0, 0.8, -1.2]))
    assert np.allclose(fit_event_model(x, y), [5.0, 0.8, -1.2])


def test_case_bends_and_beats_counterfactual():
    result = run_case("covid")
    assert result.passed, result.summary
    assert result.summary["weekly_growth_post"] < 0 < result.summary["weekly_growth_pre"]
    assert result.summary["mae_log_conditional"] < result.summary["mae_log_counterfactual"]


def test_earlier_as_of_still_runs():
    result = run_case("covid", as_of=date(2020, 3, 29))
    assert result.summary["train_weeks"] == 6
