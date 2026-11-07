"""``kairos dashboard``: serve the Streamlit app or export the static site.

The dashboard parses its own arguments (see :mod:`kairos.dashboard.cli`); this
module forwards everything after ``kairos dashboard`` to that parser.
"""

from __future__ import annotations

import typer

_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True, "help_option_names": []}


def register(root: typer.Typer) -> None:
    """Attach ``kairos dashboard`` to the root CLI."""

    @root.command(
        "dashboard",
        context_settings=_PASSTHROUGH,
        help="Dashboard: 'serve' the Streamlit app or 'export' the static site.",
    )
    def dashboard(ctx: typer.Context) -> None:
        from kairos.dashboard.cli import main

        raise typer.Exit(main(list(ctx.args)))
