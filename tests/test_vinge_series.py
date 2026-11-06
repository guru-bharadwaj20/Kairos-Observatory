import numpy as np
import pandas as pd
import pytest

from kairos.vinge.baselines import LogDriftForecaster
from kairos.vinge.series import HSTAR_COLUMNS, HorizonSpec, compute_hstar_series, hstar_series


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CACHE_DIR", str(tmp_path / "cache"))


def synthetic_bt(n_cutoffs=60, horizon=24, start_scale=0.02, end_scale=0.3):
    """Synthetic forecasts whose error per month grows over time (shrinking h*)."""
    rows = []
    for i in range(n_cutoffs):
        cutoff = pd.Timestamp("2010-01-01") + pd.DateOffset(months=i)
        rate = start_scale + (end_scale - start_scale) * i / (n_cutoffs - 1)
        for h in range(1, horizon + 1):
            err = rate * h
            rows.append(
                {
                    "series_id": "a",
                    "cutoff": cutoff,
                    "ds": cutoff + pd.DateOffset(months=h),
                    "horizon": h,
                    "y": 100.0,
                    "yhat": 100.0 * np.exp(err),
                    "yhat_lo": 100.0 * np.exp(err - 0.1 * h),
                    "yhat_hi": 100.0 * np.exp(err + 0.1 * h),
                }
            )
    return pd.DataFrame(rows)


def test_hstar_shrinks_when_error_grows():
    out = hstar_series(synthetic_bt(), HorizonSpec(window=12, min_cutoffs=6), n_jobs=1)
    assert tuple(out.columns) == HSTAR_COLUMNS
    assert out["n_cutoffs"].max() <= 12
    early = out.loc[~out["censored"]].iloc[0]["h_star"]
    late = out.iloc[-1]["h_star"]
    assert late < early
    assert late <= 3  # ln 2 / 0.3 ≈ 2.3 months


def test_no_future_information_used():
    bt = synthetic_bt(n_cutoffs=30)
    t = pd.Timestamp("2011-06-01")
    base = hstar_series(bt, HorizonSpec(window=12), eval_dates=[t], n_jobs=1)
    # Corrupting everything realised after t must not change the reading at t.
    corrupt = bt.copy()
    corrupt.loc[corrupt["ds"] > t, "yhat"] *= 1e6
    after = hstar_series(corrupt, HorizonSpec(window=12), eval_dates=[t], n_jobs=1)
    pd.testing.assert_frame_equal(base, after)
    assert base["max_horizon"].iloc[0] <= 17


def test_min_cutoffs_skips_early_months():
    out = hstar_series(synthetic_bt(n_cutoffs=10), HorizonSpec(min_cutoffs=8), n_jobs=1)
    assert (out["n_cutoffs"] >= 8).all()


def test_compute_end_to_end_with_baseline():
    rng = np.random.default_rng(3)
    n = 84
    ds = pd.date_range("2015-01-01", periods=n, freq="MS")
    logy = 3 + 0.02 * np.arange(n) + rng.normal(0, 0.05, n).cumsum()
    frame = pd.DataFrame({"ds": ds, "y": np.exp(logy), "series_id": "s"})
    out = compute_hstar_series(
        frame,
        LogDriftForecaster(),
        12,
        HorizonSpec(window=12, min_cutoffs=4),
        initial=48,
        n_jobs=1,
    )
    assert len(out) > 0
    assert out["h_star"].between(0, 12).all()
