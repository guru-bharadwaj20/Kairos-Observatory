"""Consistency checks on the paper sources (the PDF is built in CI)."""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"


def test_every_citation_is_in_the_bibliography() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex) for k in group.split(",")}
    assert cited
    assert cited <= keys, cited - keys


def test_every_result_macro_is_defined() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    results = (PAPER / "results.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"\\(res[A-Z]\w*)", tex))
    defined = set(re.findall(r"\\newcommand\{\\(res\w+)\}", results))
    assert used
    assert used <= defined, used - defined


def test_title_matches_roadmap() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert "Measuring the Distance to a Singularity" in tex
    assert "Forecastability as an Early-Warning Signal" in tex
