"""CLI subcommands, auto-discovered by :mod:`kairos.cli`.

Add a module ``kairos/commands/<name>.py`` that exposes either

* ``app: typer.Typer`` -- mounted as ``kairos <name>`` (underscores become dashes), or
* ``register(root: typer.Typer) -> None`` -- called with the root app so the module
  can attach plain commands itself.

No edit to ``cli.py`` is needed.
"""
