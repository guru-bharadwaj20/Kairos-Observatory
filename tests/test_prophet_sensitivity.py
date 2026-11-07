import numpy as np
import pytest
from typer.testing import CliRunner

from kairos.cli import app
from kairos.models.sensitivity import prior_sensitivity, stability
from kairos.models.synthetic import piecewise_exponential


def test_stability_mask():
    values = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    mask = stability(values, tol=0.1)
    assert mask.tolist() == [[True, False, False], [True, True, False]]


def test_sweep_reports_stable_region():
    frame = piecewise_exponential(
        "2010-01-01", "2020-12-01", breaks=("2015-01-01",), rates=(0.3, 0.8), noise=0.02
    )
    report = prior_sensitivity(
        frame, prior_scales=(0.05, 0.3, 1.0), n_changepoints=(10, 25), n_jobs=1, seed=1
    )
    assert len(report.grid) == 6
    assert {"stable", "in_region", "final_rate"} <= set(report.grid.columns)
    region = report.regions.iloc[0]
    assert region["region_cells"] >= 2
    assert region["recommended_prior_scale"] in (0.05, 0.3, 1.0)
    assert region["metric_min"] == pytest.approx(0.8, abs=0.15)
    md = report.to_markdown()
    assert "| synthetic |" in md


def test_bad_metric():
    with pytest.raises(ValueError, match="metric"):
        prior_sensitivity(piecewise_exponential(), metric="nope")


def test_cli_sweep(tmp_path):
    path = tmp_path / "s.csv"
    piecewise_exponential("2012-01-01", "2020-12-01", noise=0.02).to_csv(path, index=False)
    out = tmp_path / "out"
    args = ["prophet", "sweep", str(path), "-o", str(out), "-j", "1"]
    args += ["--prior-scales", "0.1,0.3", "--n-changepoints", "10,20"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (out / "sensitivity.csv").is_file()
    assert "Prior sensitivity" in (out / "sensitivity.md").read_text()
