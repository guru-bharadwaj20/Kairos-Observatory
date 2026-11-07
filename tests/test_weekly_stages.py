"""End-to-end weekly stages on the committed snapshots (offline, κ stubbed)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from kairos import weekly
from kairos.commands import run
from kairos.commands.run import RunContext, StageStatus, run_pipeline
from kairos.dashboard.snapshot import load_snapshot
from kairos.types import DialReading


def _fake_kappa(ctx: RunContext, datasets: Any) -> DialReading:
    return DialReading(
        dial="recursion", value=0.1, level=0, ci_lo=0.0, ci_hi=0.2, as_of=date(2026, 1, 1)
    )


def _no_data(ctx: RunContext, datasets: Any) -> DialReading:
    raise ValueError("not enough history")


def test_entry_points_resolve() -> None:
    for stage in run.WEEKLY_STAGES:
        assert run.resolve_entry_point(stage.entry_points[0]) is not None, stage.name


def test_weekly_pipeline_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAIROS_OFFLINE", "1")
    monkeypatch.setitem(weekly.DIAL_FUNCS, "recursion", _fake_kappa)
    monkeypatch.setitem(weekly.DIAL_FUNCS, "vinge", _no_data)
    ctx = RunContext("test", datetime(2026, 11, 7, tzinfo=UTC), tmp_path, cpu_only=True)
    report = run_pipeline(ctx)
    assert [r.status for r in report.results] == [StageStatus.OK] * 6, report.to_dict()
    assert "vinge skipped: not enough history" in report.results[2].detail

    snap = load_snapshot(tmp_path / "build" / "snapshot.json")
    assert not snap.is_demo
    assert (tmp_path / "build" / "site" / "index.html").exists()
    raw = json.loads((tmp_path / "build" / "snapshot.json").read_text("utf-8"))
    assert raw["dials"][2]["available"] is False
    assert len(snap.ledger) == 1

    # A rerun on the same data does not register the forecast twice.
    again = run_pipeline(ctx, only=["fit", "dials", "governor", "ledger"])
    assert "already registered" in again.results[4].detail
