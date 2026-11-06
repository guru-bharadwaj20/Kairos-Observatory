import json

import pytest
from typer.testing import CliRunner

from kairos.cli import app
from kairos.commands import doctor as doctor_mod
from kairos.commands.doctor import Check, run_checks
from kairos.compute import hardware

runner = CliRunner()


def test_cpu_info_is_consistent():
    cpu = hardware.cpu_info()
    assert cpu.logical_cores >= cpu.physical_cores >= 1
    if cpu.is_hybrid:
        assert cpu.performance_cores + cpu.efficiency_cores == cpu.physical_cores


def test_ram_positive():
    assert hardware.ram_bytes() > 0


def test_stan_info_finds_bundled_cmdstan():
    info = hardware.stan_info()
    assert info.ok, info.error
    assert info.model_file


def test_run_checks_cpu_only_has_no_required_failures():
    checks = run_checks(require_gpu=False)
    names = {c.name for c in checks}
    assert {"python", "numpy", "cpu", "ram", "torch", "prophet", "cmdstan"} <= names
    assert not [c for c in checks if c.required and not c.ok]


def test_json_output_cpu_only():
    result = runner.invoke(app, ["doctor", "--cpu-only", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True


def test_exit_code_nonzero_when_required_check_fails(monkeypatch):
    def fake(**_):
        return [Check("python", True, "3.12"), Check("cmdstan", False, "missing")]

    monkeypatch.setattr(doctor_mod, "run_checks", fake)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "MISSING" in result.stdout


def test_optional_failure_does_not_fail(monkeypatch):
    def fake(**_):
        return [Check("cuda", False, "no gpu", required=False)]

    monkeypatch.setattr(doctor_mod, "run_checks", fake)
    assert runner.invoke(app, ["doctor"]).exit_code == 0


@pytest.mark.gpu
def test_full_doctor_passes_on_gpu_machine():
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.stdout


@pytest.mark.slow
def test_prophet_fit_check():
    checks = run_checks(require_gpu=False, fit_prophet=True)
    fit = next(c for c in checks if c.name == "prophet fit")
    assert fit.ok, fit.detail
