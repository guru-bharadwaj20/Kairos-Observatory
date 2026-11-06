import importlib
import sys
import textwrap

import typer
from typer.testing import CliRunner

from kairos import __version__
from kairos.cli import app, command_name, discover_commands

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_command_name_dashes():
    assert command_name("fit_cache") == "fit-cache"


def test_discovers_app_and_register_modules(tmp_path, monkeypatch):
    pkg = tmp_path / "fakecmds"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "hello_world.py").write_text(
        textwrap.dedent(
            """
            import typer
            app = typer.Typer()

            @app.command()
            def greet():
                typer.echo("hi")

            @app.command()
            def other():
                pass
            """
        )
    )
    (pkg / "plain.py").write_text(
        textwrap.dedent(
            """
            def register(root):
                @root.command("ping")
                def ping():
                    print("pong")
            """
        )
    )
    (pkg / "ignored.py").write_text("x = 1\n")
    (pkg / "_private.py").write_text("raise RuntimeError('must not import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fakecmds", None)
    package = importlib.import_module("fakecmds")

    root = typer.Typer()

    @root.callback()
    def _cb():
        pass

    attached = discover_commands(root, package)
    assert attached == ["hello_world", "plain"]
    assert runner.invoke(root, ["hello-world", "greet"]).stdout.strip() == "hi"
    assert runner.invoke(root, ["ping"]).stdout.strip() == "pong"


def test_doctor_is_registered():
    result = runner.invoke(app, ["--help"])
    assert "doctor" in result.stdout
