from __future__ import annotations

from datetime import date

import pytest

from kairos.accel.bootstrap import block_bootstrap
from kairos.accel.dial import AccelerationDialConfig, acceleration_dial
from kairos.accel.synthetic import exponential_series, hyperbolic_series

AS_OF = date(2026, 11, 1)


@pytest.fixture(scope="module")
def hyper():
    out = {}
    for i in range(3):
        s = hyperbolic_series(180, delta=0.5, t_c=2017.0, noise=0.05, seed=i)
        out[f"h{i}"] = block_bootstrap(s.t, s.log_y, n_boot=199, seed=i, n_jobs=1)
    return out


@pytest.fixture(scope="module")
def expo():
    out = {}
    for i in range(2):
        s = exponential_series(180, noise=0.05, seed=20 + i)
        out[f"e{i}"] = block_bootstrap(s.t, s.log_y, n_boot=199, seed=i, n_jobs=1)
    return out


def test_normal_when_nothing_accelerates(expo):
    r = acceleration_dial(expo, {"e0": "a", "e1": "b"}, as_of=AS_OF)
    assert r.dial == "acceleration"
    assert r.detail["state"] in {"normal", "watch"}
    assert r.level <= 1
    assert r.ci_lo <= r.ci_hi


def test_single_group_is_only_watch(hyper):
    r = acceleration_dial(hyper, dict.fromkeys(hyper, "epoch"), as_of=AS_OF)
    assert r.level == 1
    assert r.detail["n_groups"] == 1
    assert r.value == pytest.approx(0.5, abs=0.1)


def test_three_groups_confirm_and_narrowing_escalates(hyper):
    groups = {"h0": "a", "h1": "b", "h2": "c"}
    first = acceleration_dial(hyper, groups, as_of=AS_OF)
    assert first.level == 2
    assert first.detail["state"] == "confirmed"
    lo, hi = first.detail["t_c_ci"]
    assert lo < 2017.0 < hi
    wide = first.model_copy(update={"detail": {**first.detail, "t_c_width": 10 * (hi - lo)}})
    second = acceleration_dial(hyper, groups, as_of=AS_OF, previous=wide)
    assert second.detail["t_c_narrowing"]
    assert second.level == 3
    third = acceleration_dial(hyper, groups, as_of=AS_OF, previous=wide, lppls_confidence=0.8)
    assert third.level == 4
    assert third.detail["state"] == "kairos"


def test_config_threshold_and_errors(hyper):
    cfg = AccelerationDialConfig(min_groups=4)
    r = acceleration_dial(hyper, {"h0": "a", "h1": "b", "h2": "c"}, as_of=AS_OF, config=cfg)
    assert r.level == 1
    with pytest.raises(ValueError, match="metric"):
        acceleration_dial({}, {}, as_of=AS_OF)
