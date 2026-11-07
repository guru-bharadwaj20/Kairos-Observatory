import json
from datetime import date

import pytest
from typer.testing import CliRunner

from kairos.cli import app
from kairos.tripwires.aggregator import Independence, load_tripwire_spec
from kairos.tripwires.lob import LobIndependenceError, enforce, independence_report
from kairos.types import DialReading


def reading(dial, sources):
    return DialReading(
        dial=dial,
        value=0.0,
        level=0,
        ci_lo=0,
        ci_hi=0,
        as_of=date(2026, 11, 1),
        detail={"sources": sources},
    )


def src(group, lab=False, name=None):
    return {"source": name or group, "source_group": group, "lab_controlled": lab}


def test_independent_dials_pass():
    rs = [
        reading("acceleration", [src("epoch"), src("labA", lab=True)]),
        reading("vinge", [src("metr")]),
    ]
    report = enforce(independence_report(rs, load_tripwire_spec().independence))
    assert report.passed


def test_single_lab_source_fails():
    rs = [reading("recursion", [src("labA", True, "a1"), src("labA", True, "a2")])]
    report = independence_report(rs)
    assert not report.passed
    assert "lab-controlled" in report.failures[0].reason
    with pytest.raises(LobIndependenceError, match="recursion"):
        enforce(report)


def test_lab_group_from_spec():
    cfg = Independence(lab_controlled_groups=["labB"])
    assert not independence_report([reading("jonas", [src("labB")])], cfg).passed


def test_missing_provenance_fails():
    report = independence_report([reading("jonas", [])])
    assert report.failures[0].reason.startswith("no provenance")


def test_cli_exit_code(tmp_path):
    path = tmp_path / "r.json"
    rs = [reading("jonas", [src("labA", True)])]
    path.write_text(json.dumps([r.model_dump(mode="json") for r in rs]), encoding="utf-8")
    assert CliRunner().invoke(app, ["governor", "lob", str(path)]).exit_code == 1
    rs = [reading("jonas", [src("labA", True), src("oecd")])]
    path.write_text(json.dumps([r.model_dump(mode="json") for r in rs]), encoding="utf-8")
    assert CliRunner().invoke(app, ["governor", "lob", str(path)]).exit_code == 0
