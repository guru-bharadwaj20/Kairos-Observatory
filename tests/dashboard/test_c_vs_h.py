from __future__ import annotations

from dataclasses import replace

from kairos.dashboard import figures
from kairos.dashboard.snapshot import SeriesFrame, load_demo
from kairos.dashboard.theme import CAPABILITY_COLOR, HUMAN_COLOR, level


def test_both_curves_always_shown() -> None:
    snap = load_demo()
    fig = figures.capability_vs_human(snap)
    names = [tr.name for tr in fig.data]
    assert snap.capability.name in names
    assert snap.human_capacity.name in names
    colors = {tr.name: tr.line.color for tr in fig.data}
    assert colors[snap.capability.name] == CAPABILITY_COLOR
    assert colors[snap.human_capacity.name] == HUMAN_COLOR


def test_response_in_title() -> None:
    snap = load_demo()
    fig = figures.capability_vs_human(snap)
    assert level(snap.alert.level).response in fig.layout.title.text


def test_gap_aligns_on_common_dates() -> None:
    snap = load_demo()
    snap = replace(
        snap,
        capability=SeriesFrame("C", "", ("2020-01-01", "2020-02-01"), (3.0, 4.0)),
        human_capacity=SeriesFrame("H", "", ("2020-02-01",), (1.5,)),
    )
    gap = figures.capability_vs_human(snap).data[2]
    assert list(gap.x) == ["2020-02-01"]
    assert list(gap.y) == [2.5]
