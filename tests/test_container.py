"""Static checks on the replication container and the validation pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kairos.commands import run

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_is_cuda_python312_and_locked() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "nvidia/cuda:12." in text
    assert "uv python install 3.12" in text
    assert "uv sync --frozen" in text
    assert "USER kairos" in text
    assert "kairos-reproduce" in text


def test_reproduce_script_runs_strict_validation() -> None:
    text = (ROOT / "docker" / "reproduce.sh").read_text(encoding="utf-8")
    assert "validation --strict" in text
    assert "kairos.commands.run" in text
    assert "docs/validation.md" in text


def test_validation_pipeline_registered() -> None:
    names = [s.name for s in run.pipelines()["validation"]]
    assert names == ["backtests", "report"]
    report = run.run_pipeline(
        run.RunContext("t", datetime.now(UTC), ROOT, dry_run=True),
        (run.Stage("x", ("kairos_absent_v:run_stage",), ""),),
    )
    assert report.unavailable == ("x",)


def test_cli_validation_strict_fails_when_stages_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run, "VALIDATION_STAGES", (run.Stage("backtests", ("kairos_absent_b:run_stage",), ""),)
    )
    assert run.main(["validation", "--strict", "--root", str(ROOT)]) == 2
