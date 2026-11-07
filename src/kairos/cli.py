"""Command-line entry point for KAIROS.

Subcommands are discovered at import time from :mod:`kairos.commands`; see that
package's docstring for the plug-in contract.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType

import typer

from kairos import __version__

app = typer.Typer(
    name="kairos",
    help="KAIROS Observatory: an early-warning system for runaway AI.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"kairos {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """KAIROS recommends; humans decide."""


def command_name(module_name: str) -> str:
    """Return the CLI name for a command module (``fit_cache`` -> ``fit-cache``)."""
    return module_name.replace("_", "-")


def discover_commands(root: typer.Typer, package: ModuleType | None = None) -> list[str]:
    """Attach every command module in ``package`` (default :mod:`kairos.commands`) to ``root``.

    Returns the names of the modules that were attached, in sorted order.
    """
    if package is None:
        package = importlib.import_module("kairos.commands")
    attached: list[str] = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        sub = getattr(module, "app", None)
        if isinstance(sub, typer.Typer):
            root.add_typer(sub, name=command_name(info.name))
            attached.append(info.name)
        elif callable(getattr(module, "register", None)):
            module.register(root)
            attached.append(info.name)
    return attached


discover_commands(app)


def main() -> None:
    """Console-script entry point."""
    for stream in (sys.stdout, sys.stderr):
        # Windows consoles default to a legacy code page that cannot print symbols such as δ.
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
