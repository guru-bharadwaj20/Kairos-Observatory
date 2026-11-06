"""Headless app tests through Streamlit's AppTest (no browser)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from kairos.dashboard.cli import APP_PATH, SNAPSHOT_ENV  # noqa: E402
from kairos.dashboard.pages import PAGES  # noqa: E402


def _run(page: str | None = None) -> Any:
    at = st_testing.AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    if page is not None:
        at.sidebar.radio(key="page").set_value(page).run()
    return at


def test_app_renders_demo() -> None:
    at = _run()
    assert not at.exception
    assert at.title[0].value == "KAIROS Observatory"
    assert any("DEMO DATA" in w.value for w in at.warning)


@pytest.mark.parametrize("page", list(PAGES))
def test_every_page_renders(page: str) -> None:
    at = _run(page)
    assert not at.exception, at.exception


def test_bad_snapshot_shows_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(SNAPSHOT_ENV, str(bad))
    at = _run()
    assert any("published snapshot" in e.value for e in at.error)
