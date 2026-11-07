from __future__ import annotations

import dataclasses
import math

import pytest

from kairos.accel.lppls import LPPLSFit
from kairos.accel.lppls_filter import LPPLSFilter, qualify


def _fit(**kw):
    base = {
        "tc": 2020.5,
        "m": 0.5,
        "omega": 8.0,
        "a": 5.0,
        "b": -1.0,
        "c1": 0.05,
        "c2": 0.05,
        "sse": 0.01,
        "t1": 2010.0,
        "t2": 2020.0,
        "n": 120,
    }
    base.update(kw)
    return LPPLSFit(**base)


def test_textbook_bubble_qualifies():
    fit = _fit()
    assert fit.damping == pytest.approx(0.5 / (8 * math.hypot(0.05, 0.05)))
    q = qualify(fit)
    assert q.qualified, q.failed


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        ({"m": 0.05}, "m"),
        ({"m": 0.95}, "m"),
        ({"omega": 4.0}, "omega"),
        ({"omega": 14.0}, "omega"),
        ({"c1": 1.0, "c2": 1.0}, "damping"),
        ({"tc": 2025.0}, "tc"),
        ({"b": 1.0}, "b"),
    ],
)
def test_each_condition_rejects(change, failed):
    q = qualify(_fit(**change))
    assert not q.qualified
    assert failed in q.failed


def test_too_few_oscillations():
    fit = _fit(omega=6.0, t1=2019.0)
    assert fit.n_oscillations < 2.5
    assert "oscillations" in qualify(fit).failed


def test_custom_filter_and_rmse_cap():
    flt = dataclasses.replace(LPPLSFilter(), max_rmse=0.001, require_negative_b=False)
    q = qualify(_fit(b=1.0), flt)
    assert "b" not in q.checks
    assert q.failed == ["rmse"]
