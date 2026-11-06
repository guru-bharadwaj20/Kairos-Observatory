import numpy as np
import pandas as pd
import pytest

from kairos.philosophy.jonas import load_policy
from kairos.tripwires.goodhart import GoodhartConfig, detect_divergence

CFG = GoodhartConfig.model_validate(load_policy().section("goodhart"))


def series(rate_per_year, n=48, noise=0.02, seed=0, sid="s"):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 12
    return pd.DataFrame(
        {
            "ds": pd.date_range("2022-01-01", periods=n, freq="MS"),
            "y": np.exp(rate_per_year * t + rng.normal(0, noise, n)),
            "series_id": sid,
        }
    )


def test_gaming_flagged():
    res = detect_divergence(series(0.8, seed=1), series(0.3, seed=2), CFG)
    assert res.flagged
    assert res.gap == pytest.approx(0.5, abs=0.05)
    reading = res.to_reading()
    assert reading.dial == "goodhart"
    assert reading.level == 1
    assert reading.ci_lo < reading.value < reading.ci_hi


def test_parallel_trends_not_flagged():
    res = detect_divergence(series(0.5, seed=1), series(0.5, seed=2), CFG)
    assert not res.flagged
    assert res.to_reading().level == 0


def test_heldout_faster_not_flagged():
    assert not detect_divergence(series(0.2, seed=1), series(0.9, seed=2), CFG).flagged


def test_small_gap_below_min_gap():
    res = detect_divergence(series(0.52, noise=1e-4), series(0.5, noise=1e-4, seed=3), CFG)
    assert res.p_value < CFG.alpha
    assert not res.flagged


def test_overlap_required():
    with pytest.raises(ValueError, match="shared months"):
        detect_divergence(series(0.5, n=6), series(0.5, n=6), CFG)
