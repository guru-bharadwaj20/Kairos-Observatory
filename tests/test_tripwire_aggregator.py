import json
from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from kairos.cli import app
from kairos.philosophy.jonas import load_policy
from kairos.tripwires.aggregator import Governor, level_for, load_tripwire_spec
from kairos.types import DialReading

DIALS = ["acceleration", "recursion", "vinge", "jonas"]
SIGN = {"acceleration": 1, "recursion": 1, "vinge": -1, "jonas": 1}
AS_OF = date(2026, 11, 1)


def reading(dial, p, groups=None, as_of=AS_OF):
    """Reading whose draws put share `p` on the danger side."""
    n = 100
    k = round(p * n)
    s = SIGN[dial]
    samples = [s * 1.0] * k + [-s * 1.0] * (n - k)
    detail = {"samples": samples}
    if groups is not None:
        detail["n_independent_groups"] = groups
    return DialReading(dial=dial, value=0.0, level=0, ci_lo=-1, ci_hi=1, as_of=as_of, detail=detail)


def governor():
    pol = load_policy()
    pol = pol.model_copy(update={"jonas": pol.jonas.model_copy(update={"dwell": 1})})
    return Governor(policy=pol)


# --- level_for: every boundary of section 8 ---------------------------------


@pytest.mark.parametrize(
    ("n", "c", "level"),
    [(0, 0, 0), (1, 0, 1), (1, 1, 2), (2, 0, 2), (2, 2, 2), (3, 0, 3), (3, 1, 3), (4, 0, 4)],
)
def test_level_table(n, c, level):
    assert level_for(n, c) == level


@given(st.integers(0, 4), st.data())
def test_level_properties(n, data):
    c = data.draw(st.integers(0, n))
    lv = level_for(n, c)
    assert 0 <= lv <= 4
    assert (lv == 0) == (n == 0)
    assert (lv == 4) == (n == 4)
    assert lv >= n
    assert lv <= max(n, 2) if n <= 2 else lv == n


@given(st.integers(0, 3), st.integers(0, 3))
def test_level_monotone(n, c):
    c = min(c, n)
    assert level_for(n + 1, c) >= level_for(n, c)
    if c < n:
        assert level_for(n, c + 1) >= level_for(n, c)


@pytest.mark.parametrize(("n", "c"), [(-1, 0), (5, 0), (1, 2)])
def test_level_invalid(n, c):
    with pytest.raises(ValueError, match="need"):
        level_for(n, c)


# --- spec -------------------------------------------------------------------


def test_spec_levels_complete():
    spec = load_tripwire_spec()
    assert [lv.name for lv in spec.levels] == ["Normal", "Watch", "Warning", "Alarm", "Kairos"]
    assert spec.dials == DIALS


def test_spec_rejects_missing_level(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        'version: "x"\ndials: [a, b, c, d]\nlevels:\n'
        "  - {level: 0, name: N, trigger: t, response: r}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="each level"):
        load_tripwire_spec(p)


# --- governor ---------------------------------------------------------------


@given(st.lists(st.booleans(), min_size=4, max_size=4), st.integers(0, 3))
def test_governor_counts_dials(danger, confirmed_idx):
    gov = governor()
    thr = gov.policy.jonas.threshold
    rs = [
        reading(d, 0.9 if hot else 0.0, groups=3 if i == confirmed_idx else 1)
        for i, (d, hot) in enumerate(zip(DIALS, danger, strict=True))
    ]
    alert = gov.evaluate(rs)
    n = sum(danger)
    c = int(danger[confirmed_idx])
    assert alert.level == level_for(n, c)
    assert alert.n_danger == n
    assert alert.threshold == pytest.approx(thr)
    assert alert.name == gov.spec.level(alert.level).name


def test_jonas_threshold_boundary():
    gov = governor()
    thr = gov.policy.jonas.threshold  # 0.10 with lambda 9
    assert gov.evaluate([reading("jonas", thr)]).level == 1  # >= enter edge
    gov = governor()
    assert gov.evaluate([reading("jonas", thr - 0.01)]).level == 0


def test_hysteresis_holds_level():
    gov = governor()
    thr = gov.policy.jonas.threshold
    band = gov.policy.jonas.band
    assert gov.evaluate([reading("jonas", 0.5)]).level == 1
    alert = gov.evaluate([reading("jonas", thr - band / 2)])
    assert alert.level == 1  # inside the band: still in danger
    assert gov.evaluate([reading("jonas", 0.0)]).level == 0


def test_rawls_worst_scenario_drives_decision():
    r = DialReading(
        dial="jonas",
        value=0.0,
        level=0,
        ci_lo=0,
        ci_hi=0,
        as_of=AS_OF,
        detail={"scenarios": {"calm": [-1.0] * 20, "hot": [1.0] * 10 + [-1.0] * 10}},
    )
    a = governor().evaluate([r]).dials[-1]
    assert a.worst_scenario == "hot"
    assert a.p_danger == pytest.approx(0.5)
    assert a.p_expected == pytest.approx(0.25)
    assert a.in_danger


def test_missing_and_ignored():
    goodhart = reading("jonas", 0.0).model_copy(update={"dial": "goodhart"})
    alert = governor().evaluate([reading("vinge", 0.9), goodhart])
    assert alert.missing == ["acceleration", "recursion", "jonas"]
    assert alert.ignored == ["goodhart"]
    assert alert.level == 1


def test_duplicate_dial_rejected():
    with pytest.raises(ValueError, match="two readings"):
        governor().evaluate([reading("jonas", 0.1), reading("jonas", 0.2)])


def test_point_estimate_needs_fallback():
    r = DialReading(dial="jonas", value=1.0, level=0, ci_lo=0.5, ci_hi=1.5, as_of=AS_OF)
    with pytest.raises(ValueError, match="point estimate"):
        governor().evaluate([r])
    gov = Governor(allow_normal_fallback=True)
    assert gov.evaluate([r]).dials[-1].method == "normal-ci"


def test_cli_evaluate(tmp_path):
    rs = [reading(d, 0.9, as_of=date(2026, 10, 1)) for d in DIALS]
    rs += [reading(d, 0.9) for d in DIALS]
    path = tmp_path / "r.json"
    path.write_text(json.dumps([r.model_dump(mode="json") for r in rs]), encoding="utf-8")
    out = CliRunner().invoke(app, ["governor", "evaluate", str(path), "--history"])
    assert out.exit_code == 0, out.output
    text = out.stdout
    alerts = json.loads(text[text.index("[\n") :])  # log lines may precede the JSON
    assert [a["level"] for a in alerts] == [0, 4]  # dwell 2 in the published policy


def test_cli_spec():
    out = CliRunner().invoke(app, ["governor", "spec"])
    assert out.exit_code == 0
    assert "4 Kairos" in out.output
