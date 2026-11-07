from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kairos.commands.run import VALIDATION_STAGES, RunContext, StageStatus, run_pipeline
from kairos.validation import runner
from kairos.validation.cases import CaseResult


def _fake_case(name: str) -> CaseResult:
    return CaseResult(name, True, {"lead": 1.0}, "always passes")


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunContext:
    monkeypatch.setattr(runner, "run_case", _fake_case)
    monkeypatch.setattr(runner, "list_cases", lambda: ["fake"])
    return RunContext(run_id="abc", as_of=datetime(2026, 1, 1, tzinfo=UTC), root=tmp_path)


def test_backtests_stage_writes_json(ctx: RunContext) -> None:
    out = runner.run_stage(ctx)
    assert out is not None
    data = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    assert data["run_id"] == "abc"
    assert data["cases"][0]["case"] == "fake"
    assert out["passed"] == 1


def test_dry_run_writes_nothing(ctx: RunContext) -> None:
    ctx.dry_run = True
    assert runner.run_stage(ctx) == {"cases": ["fake"], "dry_run": True}
    assert runner.run_report_stage(ctx) is not None
    assert not any(ctx.root.iterdir())


def test_pipeline_runs_both_stages(ctx: RunContext) -> None:
    report = run_pipeline(ctx, VALIDATION_STAGES)
    assert [r.status for r in report.results] == [StageStatus.OK, StageStatus.OK]
    assert (ctx.root / "docs" / "validation.md").exists()
    assert (ctx.root / "artifacts" / "validation" / "abc" / "backtests.json").exists()
