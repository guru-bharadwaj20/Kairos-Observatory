from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from kairos.dashboard import cli
from kairos.dashboard.snapshot import DEMO_SNAPSHOT_PATH


def test_resolve_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(cli.SNAPSHOT_ENV, raising=False)
    assert cli.resolve_snapshot(None) == DEMO_SNAPSHOT_PATH
    monkeypatch.setenv(cli.SNAPSHOT_ENV, str(tmp_path / "x.json"))
    assert cli.resolve_snapshot(None) == tmp_path / "x.json"
    assert cli.resolve_snapshot(Path("y.json")) == Path("y.json")


def test_serve_launches_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("subprocess.call", lambda c: calls.append(c) or 0)  # type: ignore[func-returns-value]
    assert cli.main(["serve", "--snapshot", str(DEMO_SNAPSHOT_PATH), "--headless"]) == 0
    cmd = calls[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[-2:] == ["--snapshot", str(DEMO_SNAPSHOT_PATH)]
    assert "--server.headless" in cmd


def test_serve_refuses_bad_snapshot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    assert cli.main(["serve", "--snapshot", str(bad)]) == 2
    assert "expected an object" in capsys.readouterr().err


def test_register_mounts_under_kairos() -> None:
    root = argparse.ArgumentParser(prog="kairos")
    cli.register(root.add_subparsers(dest="group"))
    args = root.parse_args(["dashboard", "serve", "--port", "9000"])
    assert args.port == 9000
