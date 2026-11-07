from __future__ import annotations

from typer.testing import CliRunner

from kairos.accel.synthetic import hyperbolic_series
from kairos.commands.accel import app, read_series_frame


def test_delta_command_on_csv(tmp_path):
    s = hyperbolic_series(120, delta=0.5, t_c=2012.0, seed=1, series_id="flop")
    path = tmp_path / "frame.csv"
    s.frame.to_csv(path, index=False)
    assert read_series_frame(path)["ds"].dtype.kind == "M"
    result = CliRunner().invoke(app, ["delta", str(path), "--n-boot", "50"])
    assert result.exit_code == 0, result.output
    assert "flop" in result.output


def test_delta_command_torch_cpu(tmp_path):
    s = hyperbolic_series(120, delta=0.5, t_c=2012.0, seed=1)
    path = tmp_path / "frame.csv"
    s.frame.to_csv(path, index=False)
    result = CliRunner().invoke(app, ["delta", str(path), "--n-boot", "500", "--device", "cpu"])
    assert result.exit_code == 0, result.output
