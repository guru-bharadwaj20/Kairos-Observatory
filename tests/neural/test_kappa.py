from __future__ import annotations

import json
import math
from datetime import date

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from kairos.neural import NeuralConfig, synthetic_series
from kairos.neural.kappa import (
    DIAL,
    estimate_kappa,
    kappa_from_results,
    kappa_reading,
    kappa_tasks,
)
from kairos.neural.paired import FitResult, PairedSpec
from kairos.neural.regressors import design_frame

SPEC = PairedSpec(
    config=NeuralConfig(n_forecasts=3, epochs=30, early_stopping=False),
    self_lags=6,
    test_months=12,
)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CACHE_DIR", str(tmp_path / "cache"))


def _results(exo, own, nulls):
    out = []
    for i, (e, o) in enumerate(zip(exo, own, strict=True)):
        out += [FitResult("exo", i, -1, "", (e,)), FitResult("self", i, -1, "", (o,))]
    out += [FitResult("null", 0, j, "", (n,)) for j, n in enumerate(nulls)]
    return out


def test_kappa_math_and_pvalue():
    res = kappa_from_results(_results([1.0, 1.0], [0.5, 0.7], [1.0, 0.9, 0.95, 1.1]))
    assert res.kappa == pytest.approx(0.4)
    assert res.per_seed == pytest.approx((0.5, 0.3))
    assert res.ci_lo <= res.kappa <= res.ci_hi
    assert res.p_value == pytest.approx(1 / 5)
    assert res.excess == pytest.approx(0.4 - 0.025)
    assert not res.significant


def test_kappa_null_beating_observed_raises_pvalue():
    res = kappa_from_results(_results([1.0], [0.9], [0.5, 0.95]))
    assert res.p_value == pytest.approx(2 / 3)


def test_kappa_without_null_and_without_pairs():
    res = kappa_from_results(_results([1.0], [1.0], []))
    assert res.kappa == 0.0
    assert math.isnan(res.p_value)
    assert math.isnan(res.excess)
    with pytest.raises(ValueError, match="paired"):
        kappa_from_results([FitResult("exo", 0, -1, "", (1.0,))])


def test_kappa_tasks_layout():
    design = design_frame(synthetic_series(60, seed=0), "capability")
    tasks = kappa_tasks(design, SPEC, n_seeds=3, n_permutations=5, seed=1)
    variants = [t.variant for t in tasks]
    assert variants.count("exo") == variants.count("self") == 3
    assert variants.count("null") == 5
    assert len({t.perm_seed for t in tasks if t.variant == "null"}) == 5


def test_reading_levels():
    sig = kappa_from_results(_results([1.0] * 3, [0.5] * 3, [1.0] * 30))
    assert sig.significant
    reading = kappa_reading(sig, date(2026, 1, 1))
    assert reading.dial == DIAL
    assert reading.level == 0
    assert kappa_reading(sig, date(2026, 1, 1), rising=True).level == 1
    assert reading.detail["p_value"] == pytest.approx(1 / 31)
    with pytest.raises(ValidationError):
        reading.value = 2.0  # type: ignore[misc]


def test_estimate_kappa_small_end_to_end():
    design = design_frame(synthetic_series(84, coupling=0.9, seed=0), "capability")
    res = estimate_kappa(design, SPEC, n_seeds=2, n_permutations=2, seed=0, n_jobs=1)
    assert len(res.per_seed) == 2
    assert len(res.null_kappas) == 2
    assert math.isfinite(res.kappa)


@pytest.mark.slow
def test_kappa_separates_coupled_from_uncoupled():
    spec = PairedSpec(
        config=NeuralConfig(
            n_forecasts=3,
            epochs=300,
            batch_size=16,
            learning_rate=0.1,
            n_changepoints=0,
            early_stopping=False,
        ),
        self_lags=6,
        test_months=24,
    )
    strong = estimate_kappa(
        design_frame(synthetic_series(192, coupling=0.95, seed=1), "capability"),
        spec,
        n_seeds=8,
        n_permutations=19,
        seed=0,
    )
    none = estimate_kappa(
        design_frame(synthetic_series(192, coupling=0.0, seed=1), "capability"),
        spec,
        n_seeds=8,
        n_permutations=19,
        seed=0,
    )
    assert strong.kappa > none.kappa
    assert strong.kappa > 0


def test_cli_synthetic():
    from kairos.cli import app

    result = CliRunner().invoke(
        app,
        [
            "kappa",
            "synthetic",
            "--months",
            "72",
            "--seeds",
            "1",
            "--permutations",
            "1",
            "--epochs",
            "5",
            "--n-jobs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dial"] == "recursion"
    assert payload["ci_lo"] <= payload["ci_hi"]
