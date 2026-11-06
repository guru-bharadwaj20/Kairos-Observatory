"""Tests for the weekly orchestrator, using stub stage modules."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kairos.commands import run
from kairos.commands.run import RunContext, Stage, StageStatus, main, run_pipeline

Installer = Callable[[str, Callable[[RunContext], Mapping[str, Any] | None]], None]


@pytest.fixture
def install() -> Iterator[Installer]:
    """Register a throwaway module exposing ``run_stage`` and remove it afterwards."""
    names: list[str] = []

    def _install(name: str, fn: Callable[[RunContext], Mapping[str, Any] | None]) -> None:
        mod = types.ModuleType(name)
        mod.run_stage = fn  # type: ignore[attr-defined]
        sys.modules[name] = mod
        names.append(name)

    yield _install
    for n in names:
        sys.modules.pop(n, None)


@pytest.fixture
def absent_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CLI see stand-in stages that are never importable."""
    stages = tuple(
        Stage(s.name, (f"kairos_absent_{s.name}:run_stage",), s.description)
        for s in run.WEEKLY_STAGES
    )
    monkeypatch.setattr(run, "WEEKLY_STAGES", stages)


def _ctx(tmp_path: Path, **kw: Any) -> RunContext:
    return RunContext(run_id="t", as_of=datetime(2026, 11, 13, tzinfo=UTC), root=tmp_path, **kw)


def test_missing_stage_is_unavailable_not_failure(tmp_path: Path) -> None:
    report = run_pipeline(_ctx(tmp_path), (Stage("a", ("kairos_absent_xyz:run_stage",), ""),))
    assert report.results[0].status is StageStatus.UNAVAILABLE
    assert not report.failed
    assert report.unavailable == ("a",)


def test_missing_attribute_is_unavailable(tmp_path: Path, install: Installer) -> None:
    install("stub_attr", lambda ctx: None)
    report = run_pipeline(_ctx(tmp_path), (Stage("a", ("stub_attr:nope",), ""),))
    assert report.results[0].status is StageStatus.UNAVAILABLE


def test_stages_run_in_order_and_share_artifacts(tmp_path: Path, install: Installer) -> None:
    seen: list[str] = []

    def first(ctx: RunContext) -> Mapping[str, Any]:
        seen.append("first")
        return {"n": 1}

    def second(ctx: RunContext) -> None:
        seen.append("second")
        assert ctx.artifacts["first"]["n"] == 1

    install("stub_first", first)
    install("stub_second", second)
    stages = (
        Stage("first", ("stub_first:run_stage",), ""),
        Stage("second", ("kairos_absent_q:x", "stub_second:run_stage"), ""),
    )
    report = run_pipeline(_ctx(tmp_path), stages)
    assert seen == ["first", "second"]
    assert [r.status for r in report.results] == [StageStatus.OK, StageStatus.OK]
    assert report.results[1].entry_point == "stub_second:run_stage"


def test_failure_stops_unless_keep_going(tmp_path: Path, install: Installer) -> None:
    def boom(ctx: RunContext) -> None:
        raise RuntimeError("boom")

    install("stub_boom", boom)
    install("stub_ok", lambda ctx: None)
    stages = (Stage("a", ("stub_boom:run_stage",), ""), Stage("b", ("stub_ok:run_stage",), ""))
    report = run_pipeline(_ctx(tmp_path), stages)
    assert report.failed
    assert report.results[1].status is StageStatus.SKIPPED
    report = run_pipeline(_ctx(tmp_path), stages, keep_going=True)
    assert report.results[1].status is StageStatus.OK


def test_broken_import_is_failure_not_unavailable(tmp_path: Path) -> None:
    pkg = tmp_path / "stub_broken_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import a_dependency_that_is_missing_123\n")
    sys.path.insert(0, str(tmp_path))
    try:
        report = run_pipeline(_ctx(tmp_path), (Stage("a", ("stub_broken_pkg:run_stage",), ""),))
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("stub_broken_pkg", None)
    assert report.results[0].status is StageStatus.FAILED


def test_dry_run_does_not_call(tmp_path: Path, install: Installer) -> None:
    def never(ctx: RunContext) -> None:
        raise AssertionError("called")

    install("stub_never", never)
    ctx = _ctx(tmp_path, dry_run=True)
    report = run_pipeline(ctx, (Stage("a", ("stub_never:run_stage",), ""),))
    assert report.results[0].status is StageStatus.DRY_RUN


def test_only_skip_and_unknown(tmp_path: Path) -> None:
    stages = (Stage("a", ("kairos_absent_a:x",), ""), Stage("b", ("kairos_absent_b:x",), ""))
    report = run_pipeline(_ctx(tmp_path), stages, only=["b"])
    assert report.results[0].status is StageStatus.SKIPPED
    assert report.results[1].status is StageStatus.UNAVAILABLE
    with pytest.raises(ValueError, match="zzz"):
        run_pipeline(_ctx(tmp_path), stages, skip=["zzz"])


def test_weekly_stage_order() -> None:
    names = [s.name for s in run.WEEKLY_STAGES]
    assert names == ["refresh", "fit", "dials", "governor", "ledger", "export"]


@pytest.mark.usefixtures("absent_stages")
def test_cli_report_and_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "r" / "report.json"
    code = main(["weekly", "--cpu-only", "--root", str(tmp_path), "--report", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["stages"][0]["name"] == "refresh"
    assert data["failed"] is False
    assert main(["weekly", "--strict", "--root", str(tmp_path)]) == 2
    assert main(["weekly", "--only", "bogus"]) == 2
    assert "refresh" in capsys.readouterr().out


def test_cpu_only_device(tmp_path: Path) -> None:
    assert _ctx(tmp_path, cpu_only=True).device == "cpu"
    assert _ctx(tmp_path).device == "auto"
