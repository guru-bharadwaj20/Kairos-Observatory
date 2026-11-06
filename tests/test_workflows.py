"""Structural checks on the Phase 12 GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _load(name: str) -> dict[Any, Any]:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    # PyYAML reads the bare key ``on`` as boolean True.
    if True in data:
        data["on"] = data.pop(True)
    return data


def test_weekly_is_scheduled_cpu_only_and_commits_ledger() -> None:
    wf = _load("weekly.yml")
    assert wf["on"]["schedule"][0]["cron"] == "0 6 * * 5"
    assert "workflow_dispatch" in wf["on"]
    assert wf["permissions"] == {"contents": "write"}
    job = wf["jobs"]["weekly"]
    assert job["runs-on"] == "ubuntu-latest"
    script = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert "--cpu-only" in script
    assert "kairos.commands.run" in script
    assert "git add -- ledger/" in script


def test_gpu_workflow_targets_labelled_runner_and_skips_forks() -> None:
    wf = _load("gpu.yml")
    assert "pull_request" not in wf["on"]
    assert "pull_request_target" not in wf["on"]
    assert wf["permissions"] == {"contents": "read"}
    job = wf["jobs"]["gpu-tests"]
    assert set(job["runs-on"]) >= {"self-hosted", "gpu", "kairos"}
    script = "\n".join(str(s.get("run", "")) for s in job["steps"])
    assert "pytest -m gpu" in script
    assert "pytest -m slow" in script
