from datetime import date

import pandas as pd

from kairos.validation.cases import CaseResult
from kairos.validation.report import draw_figure, render_markdown, write_report


def test_render_lists_failures():
    ok = CaseResult("a", True, {"x": 1.23456}, "crit a", notes=["caveat a"])
    bad = CaseResult("b", False, {"y": (1.0, 2.0)}, "crit b", notes=["why b failed"])
    text = render_markdown([ok, bad], {"a": "figures/validation/a.png"}, generated=date(2026, 1, 1))
    assert "| b | FAIL | crit b |" in text
    assert "**b** (FAIL): why b failed" in text
    assert "caveat a" in text
    assert "![a](figures/validation/a.png)" in text


def test_draw_figure(tmp_path):
    table = pd.DataFrame(
        {
            "as_of": [date(2000, 1, 1), date(2001, 1, 1)],
            "t_c": [2005.0, 2006.0],
            "t_c_lo": [2004.0, 2005.0],
            "t_c_hi": [2006.0, 2007.0],
            "alarm": [False, True],
        }
    )
    path = draw_figure(CaseResult("t", True, {}, "c", table=table), tmp_path)
    assert path is not None
    assert path.exists()
    assert draw_figure(CaseResult("n", True, {}, "c"), tmp_path) is None


def test_write_report_offline_cases(tmp_path):
    out, results = write_report(tmp_path, cases=["moore", "covid"])
    text = out.read_text(encoding="utf-8")
    assert "## moore" in text
    assert "## covid" in text
    assert (tmp_path / "figures" / "validation" / "covid.png").exists()
    assert all(r.passed for r in results)
