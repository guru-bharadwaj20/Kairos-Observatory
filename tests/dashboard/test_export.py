from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from kairos.dashboard import cli, export
from kairos.dashboard.snapshot import DEMO_SNAPSHOT_PATH, load_demo


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.scripts = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)
        if d.get("id"):
            self.ids.append(str(d["id"]))
        if tag == "script" and d.get("src"):
            self.scripts += 1


@pytest.fixture(scope="module")
def page() -> str:
    return export.render_site(load_demo())


def test_page_has_every_section(page: str) -> None:
    c = _Collector()
    c.feed(page)
    for sid in ("clock", "c-vs-h", "acceleration", "vinge", "ledger", "methodology", "main"):
        assert sid in c.ids
    assert len(c.ids) == len(set(c.ids)), "duplicate element ids"


def test_page_is_self_contained(page: str) -> None:
    c = _Collector()
    c.feed(page)
    assert c.scripts == 0, "no external scripts"
    assert page.count("plotly.js v") <= 2  # the library is inlined once


def test_page_content(page: str) -> None:
    snap = load_demo()
    assert 'lang="en"' in page
    assert "DEMO DATA" in page
    assert "Lead time: 31 months" in page
    assert snap.capability.name in page
    assert snap.human_capacity.name in page
    for e in snap.ledger:
        assert e.hash in page


def test_markdown_subset() -> None:
    out = export.markdown_to_html("## T\n\n| a | b |\n|---|---|\n| **x** | <y> |\n\n- one\n- two")
    assert "<h3>T</h3>" in out
    assert "<td><strong>x</strong></td><td>&lt;y&gt;</td>" in out
    assert "<li>one</li><li>two</li>" in out


def test_export_site(tmp_path: Path) -> None:
    index = export.export_site(DEMO_SNAPSHOT_PATH, tmp_path / "site")
    assert index.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    copied = json.loads((tmp_path / "site" / "snapshot.json").read_text(encoding="utf-8"))
    assert copied["is_demo"] is True
    assert (tmp_path / "site" / ".nojekyll").exists()


def test_cli_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["export", "--snapshot", str(DEMO_SNAPSHOT_PATH), "--out", str(tmp_path)]) == 0
    assert "index.html" in capsys.readouterr().out


def test_cli_export_bad_snapshot(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli.main(["export", "--snapshot", str(bad), "--out", str(tmp_path / "o")]) == 2
