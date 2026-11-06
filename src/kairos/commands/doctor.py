"""``kairos doctor``: check that this machine can run the KAIROS pipeline.

Reports the CPU P/E-core split, RAM, the torch build, CUDA devices and the
CmdStan toolchain behind Prophet. Exits with status 1 when a required component
is missing, so it can gate scripts and CI jobs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from kairos.compute.hardware import cpu_info, cuda_info, package_version, ram_bytes, stan_info

__all__ = ["Check", "register", "run_checks"]

GIB = 1024**3
MIN_RAM_GIB = 8


@dataclass(frozen=True)
class Check:
    """One line of the doctor report.

    Attributes:
        name: Component name.
        ok: Whether the component meets the requirement.
        detail: Human-readable finding.
        required: When false, failure is a warning and does not change the exit code.
    """

    name: str
    ok: bool
    detail: str
    required: bool = True


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def run_checks(*, require_gpu: bool = True, fit_prophet: bool = False) -> list[Check]:
    """Run every probe and return the checks in display order.

    Args:
        require_gpu: Treat a missing CUDA device as a failure.
        fit_prophet: Also fit a tiny Prophet model to prove CmdStan runs end to end.
    """
    checks: list[Check] = []
    py = sys.version_info
    checks.append(
        Check("python", (py.major, py.minor) == (3, 12), f"{py.major}.{py.minor}.{py.micro}")
    )

    numpy_v = package_version("numpy")
    numpy_ok = numpy_v is not None and int(numpy_v.split(".")[0]) < 2
    checks.append(Check("numpy", numpy_ok, f"{numpy_v} (need >=1.26,<2)"))

    cpu = cpu_info()
    split = (
        f"{cpu.performance_cores} P + {cpu.efficiency_cores} E"
        if cpu.is_hybrid
        else "not hybrid / unknown"
    )
    checks.append(
        Check(
            "cpu",
            cpu.logical_cores >= 1,
            f"{cpu.model}; {cpu.physical_cores} cores ({split}), {cpu.logical_cores} threads",
        )
    )

    ram = ram_bytes()
    checks.append(Check("ram", ram >= MIN_RAM_GIB * GIB, _gib(ram)))

    cuda = cuda_info()
    torch_detail = (
        f"{cuda.torch_version} (CUDA build {cuda.torch_cuda_build or 'none: CPU-only wheel'})"
        if cuda.torch_version
        else str(cuda.error)
    )
    checks.append(Check("torch", cuda.torch_version is not None, torch_detail))
    if cuda.available and cuda.devices:
        for dev in cuda.devices:
            checks.append(
                Check(
                    f"gpu[{dev['index']}]",
                    True,
                    f"{dev['name']}, {_gib(dev['total_bytes'])}, sm_{dev['capability']}",
                    required=require_gpu,
                )
            )
    else:
        checks.append(
            Check("cuda", False, cuda.error or "torch.cuda.is_available() is False", require_gpu)
        )

    for pkg in ("prophet", "neuralprophet"):
        version = package_version(pkg)
        checks.append(Check(pkg, version is not None, version or "not installed"))

    stan = stan_info()
    checks.append(
        Check(
            "cmdstan",
            stan.ok,
            f"{stan.cmdstan_path} (model: {'ok' if stan.model_file else 'missing'})"
            if stan.ok
            else str(stan.error),
        )
    )
    if fit_prophet and stan.ok:
        checks.append(_prophet_fit_check())
    return checks


def _prophet_fit_check() -> Check:
    """Fit Prophet on 36 synthetic monthly points to exercise CmdStan."""
    import logging
    import time

    import numpy as np
    import pandas as pd

    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        from prophet import Prophet

        ds = pd.date_range("2020-01-01", periods=36, freq="MS")
        y = np.log(np.exp(0.05 * np.arange(36)) + 1.0)
        start = time.perf_counter()
        Prophet(yearly_seasonality=False).fit(pd.DataFrame({"ds": ds, "y": y}))
        return Check("prophet fit", True, f"{time.perf_counter() - start:.2f} s")
    except Exception as exc:
        return Check("prophet fit", False, f"{type(exc).__name__}: {exc}")


def _render(checks: list[Check], console: Console) -> None:
    table = Table(title="kairos doctor", show_lines=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for c in checks:
        status = (
            "[green]ok[/]" if c.ok else ("[red]MISSING[/]" if c.required else "[yellow]warn[/]")
        )
        table.add_row(c.name, status, c.detail)
    console.print(table)


def doctor(
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Do not fail when no CUDA device is present."
    ),
    fit: bool = typer.Option(False, "--fit", help="Also run a tiny Prophet fit through CmdStan."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Check CPU, RAM, CUDA, torch and CmdStan; exit 1 if anything required is missing."""
    checks = run_checks(require_gpu=not cpu_only, fit_prophet=fit)
    failed = [c for c in checks if c.required and not c.ok]
    if as_json:
        payload: dict[str, Any] = {
            "ok": not failed,
            "checks": [c.__dict__ for c in checks],
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        console = Console()
        _render(checks, console)
        if failed:
            console.print(f"[red]{len(failed)} required check(s) failed.[/]")
        else:
            console.print("[green]All required checks passed.[/]")
    if failed:
        raise typer.Exit(code=1)


def register(root: typer.Typer) -> None:
    """Attach ``doctor`` as a top-level command of ``root``."""
    root.command("doctor")(doctor)
