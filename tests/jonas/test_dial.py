from __future__ import annotations

import json

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from kairos.cli import app
from kairos.jonas import JonasParams, SustainedResult, classify, jonas_dial, load_params


@pytest.fixture
def params() -> JonasParams:
    return load_params().model_copy(update={"n_boot": 30})


def test_packaged_params() -> None:
    p = load_params()
    assert p.version >= 1
    assert p.levels["closing"] == 0
    assert p.levels["sustained_robust"] == 4


def test_params_validation(tmp_path) -> None:
    raw = load_params().model_dump()
    raw["levels"]["sustained"] = 0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="decrease"):
        load_params(path)


def test_widening_dial(frame_factory, params) -> None:
    reading = jonas_dial(frame_factory(noise=0.02, seed=1), params, seed=3, n_jobs=1)
    assert reading.dial == "jonas"
    assert reading.value == pytest.approx(0.55, abs=0.15)
    assert reading.ci_lo <= reading.value <= reading.ci_hi
    assert reading.detail["state"] == "sustained_robust"
    assert reading.level == 4
    assert reading.detail["run_length"] >= 6
    assert reading.detail["laggard"] == "h_funding"
    assert reading.as_of.isoformat() == "2024-12-01"
    json.dumps(reading.model_dump(mode="json"))


def test_closing_dial(frame_factory, params) -> None:
    frame = frame_factory({"c": 0.2}, {"h": 0.5}, noise=0.02, seed=2)
    reading = jonas_dial(frame, params, seed=3, n_jobs=1)
    assert reading.detail["state"] == "closing"
    assert reading.level == 0
    assert reading.value < 0


def test_too_short(frame_factory, params) -> None:
    with pytest.raises(ValueError, match="months"):
        jonas_dial(frame_factory(periods=8, start="2016-01-01"), params, n_jobs=1)


def _result(run: bool, hac: bool) -> SustainedResult:
    return SustainedResult(
        run_length=6 if run else 1,
        min_run=6,
        run_rule=run,
        mean_dgdt=0.1,
        se=0.05,
        t_stat=2.0,
        p_value=0.02 if hac else 0.3,
        alpha=0.05,
        hac_rule=hac,
        n=36,
        maxlags=3,
        sustained=run and hac,
    )


def test_classify() -> None:
    assert classify(-0.1, -0.2, _result(True, True)) == "closing"
    assert classify(0.1, -0.1, _result(False, False)) == "widening"
    assert classify(0.1, -0.1, _result(True, False)) == "persistent"
    assert classify(0.1, -0.1, _result(True, True)) == "sustained"
    assert classify(0.1, 0.05, _result(True, True)) == "sustained_robust"


def test_cli_dial(frame_factory, tmp_path) -> None:
    csv = tmp_path / "frame.csv"
    frame_factory(noise=0.02, seed=1).to_csv(csv, index=False)
    cfg = tmp_path / "jonas.yaml"
    cfg.write_text(yaml.safe_dump(load_params().model_dump() | {"n_boot": 10}), encoding="utf-8")
    result = CliRunner().invoke(
        app, ["jonas", "dial", str(csv), "--config", str(cfg), "--seed", "1", "--n-jobs", "1"]
    )
    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)
    assert out["dial"] == "jonas"
    assert out["detail"]["state"] in {"sustained", "sustained_robust"}
