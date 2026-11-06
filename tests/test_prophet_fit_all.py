import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from kairos.cli import app
from kairos.models.fit_all import default_series_per_task, fit_all, forecast_frame
from kairos.models.null_model import ProphetConfig
from kairos.models.synthetic import piecewise_exponential


def _panel(n: int) -> pd.DataFrame:
    parts = [
        piecewise_exponential(
            "2012-01-01",
            "2020-12-01",
            rates=(0.2 + 0.1 * i,),
            noise=0.03,
            seed=i,
            series_id=f"s{i}",
        )
        for i in range(n)
    ]
    return pd.concat(parts, ignore_index=True)


def test_default_series_per_task():
    assert default_series_per_task(1, 28) == 1
    assert default_series_per_task(1000, 28) == 9
    assert default_series_per_task(10, 1) == 3


def test_fit_all_in_process_keeps_order_and_times():
    panel = _panel(3)
    result = fit_all(panel, n_jobs=1, horizon=6, series_per_task=2, seed=1)
    assert list(result.fits) == ["s0", "s1", "s2"]
    assert result.wall_seconds > 0
    assert result.meta["n_tasks"] == 2
    summary = result.summary()
    assert list(summary["series_id"]) == ["s0", "s1", "s2"]
    assert summary["final_rate"].is_monotonic_increasing
    assert all(f.model is None for f in result.fits.values())
    fc = forecast_frame(result.fits)
    assert set(fc["series_id"]) == {"s0", "s1", "s2"}


def test_fit_all_parallel_matches_serial():
    panel = _panel(4)
    cfg = ProphetConfig(uncertainty_samples=0)
    serial = fit_all(panel, n_jobs=1, horizon=0, config=cfg, seed=2)
    parallel = fit_all(panel, n_jobs=2, horizon=0, config=cfg, seed=2)
    assert parallel.n_jobs == 2
    for sid in serial.fits:
        assert parallel.fits[sid].final_rate == pytest.approx(serial.fits[sid].final_rate)


def test_fit_all_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CACHE_DIR", str(tmp_path))
    panel = _panel(1)
    a = fit_all(panel, n_jobs=1, horizon=0, cache=True, seed=3)
    b = fit_all(panel, n_jobs=1, horizon=0, cache=True, seed=3)
    assert a.fits["s0"].fit_seconds == b.fits["s0"].fit_seconds  # loaded, not refitted


def test_fit_all_unknown_series():
    with pytest.raises(ValueError, match="not in frame"):
        fit_all(_panel(1), series_ids=["nope"], n_jobs=1)


def test_cli_prophet_fit(tmp_path):
    path = tmp_path / "series.csv"
    _panel(2).to_csv(path, index=False)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app, ["prophet", "fit", str(path), "--out", str(out), "-j", "1", "--horizon", "3"]
    )
    assert result.exit_code == 0, result.output
    assert (out / "summary.csv").is_file()
    assert json.loads((out / "run.json").read_text())["wall_seconds"] > 0
    fc = pd.read_csv(out / "forecast.csv")
    assert len(fc) == 2 * (108 + 3)
