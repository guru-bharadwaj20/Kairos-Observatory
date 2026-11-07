from __future__ import annotations

import numpy as np
import pytest

from kairos.accel.lppls import lppls_series
from kairos.accel.lppls_confidence import lppls_confidence, lppls_confidence_frame
from kairos.accel.synthetic import exponential_series

LENGTHS = (60, 84, 108)


@pytest.fixture(scope="module")
def bubble():
    return lppls_series(144, start="2008-01-01", tc=2020.3, noise=0.005, seed=2)


def test_bubble_has_high_confidence_near_the_end(bubble):
    ends = [len(bubble.t) - 1]
    out = lppls_confidence(bubble.t, bubble.log_y, ends=ends, window_lengths=LENGTHS, device="cpu")
    assert out["n_windows"].iloc[0] == 3
    assert out["confidence"].iloc[0] >= 2 / 3


def test_exponential_has_zero_confidence():
    s = exponential_series(144, noise=0.02, seed=1)
    out = lppls_confidence(s.t, s.log_y, ends=[143], window_lengths=LENGTHS, device="cpu")
    assert out["confidence"].iloc[0] == 0.0


def test_confidence_series_shape_and_short_history(bubble):
    out = lppls_confidence(
        bubble.t, bubble.log_y, ends=[50, 100, 143], window_lengths=LENGTHS, device="cpu"
    )
    assert list(out.columns) == ["t", "end", "n_windows", "n_qualified", "confidence"]
    assert np.isnan(out["confidence"].iloc[0])  # no window fits before index 59
    assert out["n_windows"].tolist() == [0, 2, 3]


def test_frame_wrapper_adds_ds(bubble):
    out = lppls_confidence_frame(bubble.frame, last=2, window_lengths=LENGTHS, device="cpu")
    assert len(out) == 2
    assert out["ds"].iloc[-1] == bubble.frame["ds"].iloc[-1]


def test_rejects_short_windows(bubble):
    with pytest.raises(ValueError, match="window lengths"):
        lppls_confidence(bubble.t, bubble.log_y, window_lengths=(3,))
