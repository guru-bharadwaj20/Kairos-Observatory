import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from kairos.cli import app
from kairos.vinge import load_thresholds, vinge_dial
from kairos.vinge.dial import VingeThresholds

runner = CliRunner()


def hstar_table(slopes, start=24.0, n=48, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    as_of = pd.date_range("2020-01-01", periods=n, freq="MS")
    frames = []
    for i, slope in enumerate(slopes):
        h = np.clip(start + slope * np.arange(n) / 12 + rng.normal(0, noise, n), 0, None)
        frames.append(
            pd.DataFrame({"series_id": f"s{i}", "as_of": as_of, "h_star": h, "censored": False})
        )
    return pd.concat(frames, ignore_index=True)


def test_packaged_thresholds_load():
    th = load_thresholds()
    assert th.version >= 1
    assert th.tolerance_factor == 2.0


def test_thresholds_validate_order(tmp_path):
    bad = load_thresholds().model_dump() | {"critical_months": 10.0}
    with pytest.raises(ValidationError, match="critical_months"):
        VingeThresholds.model_validate(bad)


def test_normal_when_flat():
    reading = vinge_dial(hstar_table([0.0, 0.0, 0.0]))
    assert reading.dial == "vinge"
    assert reading.detail["state"] == "normal"
    assert reading.level == 0
    assert reading.ci_lo <= reading.value <= reading.ci_hi
    assert reading.as_of == date(2023, 12, 1)


def test_watch_when_one_series_shrinks():
    reading = vinge_dial(hstar_table([-4.0, 0.0, 0.0]))
    assert reading.detail["state"] == "watch"
    assert reading.detail["n_shrinking"] == 1
    assert reading.detail["series"]["s0"]["shrinking"]


def test_warning_and_critical_need_confirmation():
    # Three series shrinking from 24 to ~4 months over 4 years -> warning.
    warn = vinge_dial(hstar_table([-5.0, -5.0, -5.0]))
    assert warn.detail["state"] == "warning"
    assert warn.level == 2
    crit = vinge_dial(hstar_table([-6.5, -6.5, -6.5], start=24.5, noise=0.1))
    assert crit.value <= 1.0
    assert crit.detail["state"] == "critical"
    assert crit.level == 3


def test_as_of_ignores_later_readings():
    table = hstar_table([-5.0, -5.0, -5.0])
    early = vinge_dial(table, as_of=date(2021, 6, 1))
    assert early.as_of == date(2021, 6, 1)
    assert early.value > 10
    with pytest.raises(ValueError, match="no h"):
        vinge_dial(table, as_of=date(2000, 1, 1))


def test_cli_hstar_and_dial(tmp_path):
    rows = []
    for c in range(12):
        cutoff = pd.Timestamp("2020-01-01") + pd.DateOffset(months=c)
        for h in range(1, 7):
            rows.append(
                {
                    "series_id": "a",
                    "cutoff": cutoff,
                    "ds": cutoff + pd.DateOffset(months=h),
                    "horizon": h,
                    "y": 10.0,
                    "yhat": 10.0 * np.exp(0.2 * h),
                    "yhat_lo": 5.0,
                    "yhat_hi": 40.0,
                }
            )
    bt_path = tmp_path / "bt.csv"
    pd.DataFrame(rows).to_csv(bt_path, index=False)
    hs_path = tmp_path / "hstar.csv"
    result = runner.invoke(app, ["vinge", "hstar", str(bt_path), "-o", str(hs_path)])
    assert result.exit_code == 0, result.output
    assert hs_path.exists()
    result = runner.invoke(app, ["vinge", "dial", str(hs_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dial"] == "vinge"
    assert payload["value"] == pytest.approx(3.0)  # ln2/0.2 = 3.47 -> crosses at 4
