from __future__ import annotations

import statistics
from dataclasses import replace

import pytest

from kairos.dashboard import figures
from kairos.dashboard.snapshot import load_demo


def test_delta_summary_known_values() -> None:
    s = figures.delta_summary(tuple(float(x) for x in range(-10, 91)))  # -10..90
    assert s["median"] == 40.0
    assert s["lower"] == pytest.approx(-5.0)
    assert s["upper"] == pytest.approx(85.0)
    assert s["p_positive"] == pytest.approx(90 / 101)


def test_delta_summary_matches_statistics_median() -> None:
    xs = load_demo().acceleration[0].delta_samples
    assert figures.delta_summary(xs)["median"] == pytest.approx(statistics.median(xs))


def test_delta_summary_empty() -> None:
    with pytest.raises(ValueError, match="no bootstrap"):
        figures.delta_summary(())


def test_fit_figure_marks_changepoints() -> None:
    a = load_demo().acceleration[0]
    fig = figures.acceleration_fit(a)
    vlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(vlines) == len(a.changepoints)
    assert "changepoint" in [tr.name for tr in fig.data]


def test_fit_figure_without_changepoints() -> None:
    a = replace(load_demo().acceleration[0], changepoints=())
    names = [tr.name for tr in figures.acceleration_fit(a).data]
    assert "changepoint" not in names


def test_delta_figure_title_reports_interval() -> None:
    a = load_demo().acceleration[1]
    fig = figures.delta_distribution(a)
    assert "90% interval" in fig.layout.title.text
    assert list(fig.data[0].x) == list(a.delta_samples)
