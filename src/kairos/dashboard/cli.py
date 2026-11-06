"""``kairos dashboard`` command.

The project-wide ``kairos`` CLI mounts this through :func:`register`; until it
exists, ``python -m kairos.dashboard`` gives the same commands.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from kairos.dashboard.snapshot import DEMO_SNAPSHOT_PATH, SnapshotError, load_snapshot

SNAPSHOT_ENV = "KAIROS_SNAPSHOT"
APP_PATH = Path(__file__).with_name("app.py")


def resolve_snapshot(path: Path | None) -> Path:
    """Explicit path, else ``$KAIROS_SNAPSHOT``, else the demo fixture."""
    if path is not None:
        return path
    env = os.environ.get(SNAPSHOT_ENV)
    return Path(env) if env else DEMO_SNAPSHOT_PATH


def streamlit_command(snapshot: Path, port: int, headless: bool) -> list[str]:
    """Run streamlit command."""
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP_PATH),
        "--server.port", str(port), "--browser.gatherUsageStats", "false",
    ]  # fmt: skip
    if headless:
        cmd += ["--server.headless", "true"]
    return [*cmd, "--", "--snapshot", str(snapshot)]


def _check(path: Path) -> str | None:
    try:
        load_snapshot(path)
    except (OSError, SnapshotError) as exc:
        return str(exc)
    return None


def _serve(args: argparse.Namespace) -> int:
    snap = resolve_snapshot(args.snapshot)
    err = _check(snap)  # fail fast, before starting a server
    if err:
        print(f"kairos dashboard: {err}", file=sys.stderr)
        return 2
    return subprocess.call(streamlit_command(snap, args.port, args.headless))


def build_parser(p: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Run build parser."""
    p = p or argparse.ArgumentParser(prog="kairos dashboard")
    p.set_defaults(func=_serve, snapshot=None, port=8501, headless=False)
    cmds = p.add_subparsers(dest="command")
    serve = cmds.add_parser("serve", help="start the Streamlit app (default)")
    serve.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help=f"published snapshot JSON (default: ${SNAPSHOT_ENV}, else demo)",
    )
    serve.add_argument("--port", type=int, default=8501)
    serve.add_argument("--headless", action="store_true")
    serve.set_defaults(func=_serve)
    return p


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach ``dashboard`` to the parent ``kairos`` parser."""
    build_parser(sub.add_parser("dashboard", help="run or export the public dashboard"))


def main(argv: Sequence[str] | None = None) -> int:
    """Run main."""
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result
