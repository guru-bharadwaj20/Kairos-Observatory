"""The documentation site config points only at files that exist."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


def _pages(nav: Any) -> Iterator[str]:
    if isinstance(nav, str):
        yield nav
    elif isinstance(nav, list):
        for item in nav:
            yield from _pages(item)
    elif isinstance(nav, dict):
        for value in nav.values():
            yield from _pages(value)


def test_mkdocs_nav_files_exist() -> None:
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert config["theme"]["name"] == "material"
    pages = list(_pages(config["nav"]))
    assert "index.md" in pages
    for page in pages:
        assert (ROOT / "docs" / page).is_file(), page


def test_pages_workflow_deploys_from_main_only() -> None:
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8"))
    deploy = wf["jobs"]["deploy"]
    assert "refs/heads/main" in deploy["if"]
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
