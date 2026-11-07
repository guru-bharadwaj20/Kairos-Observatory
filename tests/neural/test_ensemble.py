from __future__ import annotations

import pytest

from kairos.neural import NeuralConfig, synthetic_series
from kairos.neural.ensemble import (
    DEFAULT_N_SEEDS,
    ensemble_device,
    ensemble_seeds,
    run_seed_ensemble,
    summarize,
)
from kairos.neural.paired import FitResult, PairedSpec
from kairos.neural.regressors import design_frame

SPEC = PairedSpec(
    config=NeuralConfig(n_forecasts=2, epochs=5, early_stopping=False),
    self_lags=6,
    test_months=12,
)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CACHE_DIR", str(tmp_path / "cache"))


def _design():
    return design_frame(synthetic_series(72, seed=0), "capability")


def test_default_seed_count():
    assert DEFAULT_N_SEEDS >= 20


def test_ensemble_device_policy():
    assert ensemble_device(None) == "cpu"
    assert ensemble_device("auto") == "cpu"
    assert ensemble_device("cpu") == "cpu"


def test_ensemble_seeds_stable_and_distinct():
    a = ensemble_seeds(1, 20)
    assert a == ensemble_seeds(1, 20)
    assert len(set(a)) == 20
    assert ensemble_seeds(1, 3, "w1") != ensemble_seeds(1, 3, "w2")
    with pytest.raises(ValueError, match="n must"):
        ensemble_seeds(1, 0)


def test_summarize():
    results = [
        FitResult("exo", 1, -1, "", (1.0, 3.0)),
        FitResult("exo", 2, -1, "", (2.0, 2.0)),
        FitResult("self", 1, -1, "", (1.0, 1.0)),
    ]
    table = summarize(results).set_index("variant")
    assert table.loc["exo", "mean"] == pytest.approx(2.0)
    assert table.loc["exo", "n"] == 2
    assert table.loc["self", "std"] == 0.0


def test_seed_ensemble_in_process_and_cached():
    design = _design()
    first = run_seed_ensemble(design, SPEC, n_seeds=2, seed=3, n_jobs=1)
    assert [r.variant for r in first] == ["exo", "self", "exo", "self"]
    assert first[0].seed == first[1].seed
    again = run_seed_ensemble(design, SPEC, n_seeds=2, seed=3, n_jobs=1)
    assert [r.errors for r in again] == [r.errors for r in first]


@pytest.mark.slow
def test_seed_ensemble_process_pool():
    results = run_seed_ensemble(_design(), SPEC, n_seeds=4, seed=1, n_jobs=4, use_cache=False)
    assert len(results) == 8


@pytest.mark.gpu
def test_seed_ensemble_gpu():
    results = run_seed_ensemble(_design(), SPEC, n_seeds=2, seed=1, device="cuda", use_cache=False)
    assert len(results) == 4
