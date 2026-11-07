"""Release plumbing: changelog format and the GitHub release workflow."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


def test_changelog_has_v010_and_links() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
    assert re.search(r"^## \[0\.1\.0\] - \d{4}-\d{2}-\d{2}$", text, re.MULTILINE)
    assert "[0.1.0]: https://github.com/" in text


def test_release_publishes_github_release_without_tokens() -> None:
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8"))
    trigger = wf.get("on", wf.get(True))
    assert trigger["push"]["tags"] == ["v*.*.*"]
    jobs = wf["jobs"]
    assert "publish-pypi" not in jobs
    release = jobs["github-release"]
    assert release["needs"] == "build"
    assert release["permissions"] == {"contents": "write"}
    run = release["steps"][-1]["run"]
    assert "gh release create" in run
    assert "secrets." not in (ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8")
