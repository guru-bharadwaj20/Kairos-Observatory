from __future__ import annotations

import pytest

from kairos.dashboard import figures
from kairos.dashboard.demo import build_demo_snapshot
from kairos.dashboard.snapshot import SnapshotError, VingeHorizon, load_demo, parse_snapshot

T = ("2020-01-01", "2021-01-01", "2022-01-01")


def test_linear_trend_exact() -> None:
    a, b = figures.linear_trend(T, (30.0, 27.0, 24.0))
    assert b == pytest.approx(-3.0)
    assert a + b * 2021 == pytest.approx(27.0)


def test_linear_trend_needs_two_points() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        figures.linear_trend(T[:1], (1.0,))


def test_tolerance_lookup() -> None:
    vh = VingeHorizon(T, (3.0, 2.0, 1.0), "months", {1.5: (2.0, 1.0, 0.5)})
    assert vh.tolerances(2.0) == [1.5, 2.0]
    assert vh.at(2.0, 2.0) == (3.0, 2.0, 1.0)
    assert vh.at(1.5, 2.0) == (2.0, 1.0, 0.5)
    with pytest.raises(KeyError):
        vh.at(9.0, 2.0)


def test_figure_reports_shrinking_trend() -> None:
    snap = load_demo()
    pub = snap.parameters.vinge_tolerance_factor
    fig = figures.vinge_horizon(snap.vinge_horizon, pub, pub, trend_months=60)
    assert "shrinking" in fig.layout.title.text
    assert len(fig.data) == 2


def test_figure_switches_tolerance() -> None:
    snap = load_demo()
    vh, pub = snap.vinge_horizon, snap.parameters.vinge_tolerance_factor
    fig = figures.vinge_horizon(vh, 1.5, pub)
    assert list(fig.data[0].y) == list(vh.by_tolerance[1.5])
    assert "published tolerance is 2x" in fig.layout.title.text


def test_bad_tolerance_key() -> None:
    raw = build_demo_snapshot()
    raw["vinge_horizon"]["by_tolerance"] = {"two": [1.0]}  # type: ignore[index]
    with pytest.raises(SnapshotError, match="not a number"):
        parse_snapshot(raw)
