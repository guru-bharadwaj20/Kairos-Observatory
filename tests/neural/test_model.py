from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError

from kairos.neural import (
    NeuralConfig,
    build_model,
    fit_model,
    predict,
    quiet_neuralprophet,
    synthetic_series,
    to_np_frame,
    trainer_config,
)

FAST = NeuralConfig(epochs=5, early_stopping=False)


def test_trainer_config_cpu():
    cfg = trainer_config("cpu")
    assert cfg["accelerator"] == "cpu"
    assert cfg["devices"] == 1
    assert cfg["enable_progress_bar"] is False
    assert cfg["enable_checkpointing"] is False


def test_trainer_config_follows_settings(monkeypatch):
    monkeypatch.setenv("KAIROS_DEVICE", "cpu")
    assert trainer_config()["accelerator"] == "cpu"


def test_trainer_config_cuda_maps_to_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cfg = trainer_config(torch.device("cuda:0"))
    assert cfg == {**cfg, "accelerator": "gpu", "devices": [0]}


def test_config_validates_quantiles():
    assert NeuralConfig(quantiles=(0.95, 0.05, 0.05)).quantiles == (0.05, 0.95)
    with pytest.raises(ValidationError):
        NeuralConfig(quantiles=(1.5,))
    with pytest.raises(ValidationError):
        NeuralConfig(n_forecasts=0)


def test_config_is_frozen_and_hashable():
    cfg = NeuralConfig()
    with pytest.raises(ValidationError):
        cfg.n_lags = 3  # type: ignore[misc]
    assert hash(cfg) == hash(NeuralConfig())


def test_quiet_restores_levels():
    logger = logging.getLogger("NP")
    logger.setLevel(logging.INFO)
    with quiet_neuralprophet():
        assert logger.level == logging.ERROR
        warnings.warn("hidden", FutureWarning, stacklevel=1)
    assert logger.level == logging.INFO


def test_to_np_frame_logs_and_sorts():
    sf = synthetic_series(24, seed=1).sample(frac=1.0, random_state=0)
    out = to_np_frame(sf, "capability")
    assert list(out.columns) == ["ds", "y"]
    assert out["ds"].is_monotonic_increasing
    raw = sf[sf.series_id == "capability"].sort_values("ds")["y"].to_numpy()
    np.testing.assert_allclose(out["y"].to_numpy(), np.log(raw))
    with pytest.raises(KeyError):
        to_np_frame(sf, "missing")


def test_synthetic_series_shape():
    sf = synthetic_series(60, seed=3)
    assert set(sf.series_id) == {"capability", "compute", "investment", "researchers"}
    assert (sf.groupby("series_id").size() == 60).all()
    with pytest.raises(ValueError, match="coupling"):
        synthetic_series(10, coupling=1.0)


def test_fit_cpu_quick(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = to_np_frame(synthetic_series(60, seed=0), "capability")
    model = build_model(FAST, "cpu")
    metrics = fit_model(model, df, config=FAST, seed=1)
    assert isinstance(metrics, pd.DataFrame)
    forecast = predict(model, df)
    assert forecast["yhat1"].notna().sum() == len(df)
    assert not (tmp_path / "lightning_logs").exists()


def test_fit_with_early_stopping_and_validation():
    df = to_np_frame(synthetic_series(72, seed=0), "capability")
    cfg = NeuralConfig(epochs=30, early_stopping=True)
    model = build_model(cfg, "cpu")
    fit_model(model, df.iloc[:60], config=cfg, validation=df.iloc[60:], seed=2)
    assert predict(model, df)["yhat1"].notna().all()


def test_fit_is_reproducible():
    df = to_np_frame(synthetic_series(48, seed=0), "capability")
    preds = []
    for _ in range(2):
        model = build_model(FAST, "cpu")
        fit_model(model, df, config=FAST, seed=7)
        preds.append(predict(model, df)["yhat1"].to_numpy())
    np.testing.assert_allclose(preds[0], preds[1], rtol=1e-6)


@pytest.mark.gpu
def test_fit_gpu():
    df = to_np_frame(synthetic_series(48, seed=0), "capability")
    model = build_model(FAST, "cuda")
    fit_model(model, df, config=FAST, seed=1)
    assert predict(model, df)["yhat1"].notna().all()
