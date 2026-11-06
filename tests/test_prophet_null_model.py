import numpy as np
import pandas as pd
import pytest

from kairos.models.null_model import (
    ProphetConfig,
    fit_null_model,
    is_frontier,
    select_series,
    series_seed,
)
from kairos.models.synthetic import piecewise_exponential


def test_config_defaults():
    cfg = ProphetConfig()
    assert cfg.interval_width == 0.9
    assert cfg.changepoint_prior_scale == 0.3
    assert cfg.prophet_kwargs(frontier=True)["yearly_seasonality"] is False
    assert cfg.prophet_kwargs(frontier=False)["yearly_seasonality"] == "auto"


def test_config_from_env_and_yaml(monkeypatch, tmp_path):
    path = tmp_path / "kairos.yaml"
    path.write_text("n_jobs: 4\nprophet:\n  changepoint_prior_scale: 0.05\n  n_changepoints: 10\n")
    monkeypatch.setenv("KAIROS_CONFIG", str(path))
    cfg = ProphetConfig()
    assert cfg.changepoint_prior_scale == 0.05
    assert cfg.n_changepoints == 10
    monkeypatch.setenv("KAIROS_PROPHET_N_CHANGEPOINTS", "7")
    assert ProphetConfig().n_changepoints == 7
    assert ProphetConfig(n_changepoints=3).n_changepoints == 3


def test_is_frontier():
    assert is_frontier("compute_frontier")
    assert not is_frontier("safety_pubs")
    assert is_frontier("x", ProphetConfig(frontier_patterns=("x",)))


def test_series_seed_stable_and_distinct():
    assert series_seed(1, "a") == series_seed(1, "a")
    assert series_seed(1, "a") != series_seed(1, "b")


def test_select_series_needs_id_for_multi():
    a = piecewise_exponential(series_id="a")
    b = piecewise_exponential(series_id="b")
    both = pd.concat([a, b], ignore_index=True)
    with pytest.raises(ValueError, match="pass series_id"):
        select_series(both)
    sid, rows = select_series(both, "b")
    assert sid == "b"
    assert len(rows) == len(b)


def test_fit_null_model_rates_and_forecast():
    frame = piecewise_exponential(
        "2005-01-01",
        "2024-12-01",
        breaks=("2015-01-01",),
        rates=(0.3, 0.9),
        noise=0.02,
        series_id="compute_frontier",
    )
    fit = fit_null_model(frame, seed=3, horizon=12)
    assert fit.frontier
    assert fit.params["yearly_seasonality"] is False
    assert fit.params["interval_width"] == 0.9
    assert len(fit.forecast) == len(frame) + 12
    assert len(fit.deltas) == len(fit.changepoints) == 25
    assert fit.base_rate == pytest.approx(0.3, abs=0.08)
    assert fit.final_rate == pytest.approx(0.9, abs=0.1)
    assert fit.rates[-1] == pytest.approx(fit.final_rate)
    assert fit.model is not None
    light = fit.without_model()
    assert light.model is None
    assert light.final_rate == fit.final_rate


def test_fit_null_model_deterministic():
    frame = piecewise_exponential("2010-01-01", "2020-12-01", rates=(0.5,), noise=0.05)
    a = fit_null_model(frame, seed=11, horizon=6)
    b = fit_null_model(frame, seed=11, horizon=6)
    np.testing.assert_allclose(a.forecast["yhat_upper"], b.forecast["yhat_upper"])
    assert a.seed == b.seed
