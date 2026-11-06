from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kairos.dashboard.demo import build_demo_snapshot
from kairos.dashboard.snapshot import (
    DEMO_SNAPSHOT_PATH,
    DIAL_ORDER,
    SnapshotError,
    load_demo,
    load_snapshot,
    parse_snapshot,
)


def test_fixture_is_current() -> None:
    on_disk = json.loads(DEMO_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert on_disk == build_demo_snapshot(), "run: python -m kairos.dashboard.demo"


def test_demo_loads() -> None:
    snap = load_demo()
    assert snap.is_demo
    assert tuple(d.dial for d in snap.dials) == DIAL_ORDER
    assert 0 <= snap.alert.level <= 4
    assert len(snap.capability.t) == len(snap.human_capacity.t)
    assert snap.ledger[1].prev_hash == snap.ledger[0].hash


def _demo() -> dict[str, object]:
    return copy.deepcopy(build_demo_snapshot())


@pytest.mark.parametrize(
    ("path", "value", "msg"),
    [
        (("schema_version",), 2, "schema_version"),
        (("alert", "level"), 5, "alert.level"),
        (("alert", "level"), 1.5, "alert.level"),
        (("dials", 0, "p_danger"), 1.2, "p_danger"),
        (("dials", 0, "state"), "warm", "state"),
        (("ledger", 0, "proof"), "maybe", "proof"),
        (("alert", "lead_time", "lower"), 99.0, "lower <= median"),
    ],
)
def test_rejects_bad_values(path: tuple[object, ...], value: object, msg: str) -> None:
    raw = _demo()
    node: object = raw
    for key in path[:-1]:
        node = node[key]  # type: ignore[index]
    node[path[-1]] = value  # type: ignore[index]
    with pytest.raises(SnapshotError, match=msg):
        parse_snapshot(raw)


def test_rejects_missing_dial() -> None:
    raw = _demo()
    raw["dials"] = raw["dials"][1:]  # type: ignore[index]
    with pytest.raises(SnapshotError, match="one reading per dial"):
        parse_snapshot(raw)


def test_rejects_ragged_columns() -> None:
    raw = _demo()
    raw["capability"]["value"] = [1.0]  # type: ignore[index]
    with pytest.raises(SnapshotError, match="different lengths"):
        parse_snapshot(raw)


def test_extra_keys_tolerated_and_no_lead_time(tmp_path: Path) -> None:
    raw = _demo()
    raw["future_field"] = {"x": 1}
    raw["alert"]["lead_time"] = None  # type: ignore[index]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    assert load_snapshot(p).alert.lead_time is None


def test_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "s.json"
    p.write_text("{", encoding="utf-8")
    with pytest.raises(SnapshotError, match="not valid JSON"):
        load_snapshot(p)
