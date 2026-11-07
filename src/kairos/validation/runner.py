"""Stage entry points for ``kairos run validation``.

``run_stage`` (the ``backtests`` stage) runs every registered historical case
and writes a JSON summary to ``<root>/artifacts/validation/<run_id>/``.
``run_report_stage`` (the ``report`` stage) draws the figures and writes
``<root>/docs/validation.md``, reusing the backtest results when present.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kairos.validation.cases import CaseResult, list_cases, run_case

if TYPE_CHECKING:
    from kairos.commands.run import RunContext

__all__ = ["artifact_dir", "run_report_stage", "run_stage"]

log = logging.getLogger("kairos.validation.runner")


def artifact_dir(ctx: RunContext) -> Path:
    """Directory for this run's validation artifacts."""
    return ctx.root / "artifacts" / "validation" / ctx.run_id


def _case_dict(result: CaseResult) -> dict[str, Any]:
    return {
        "case": result.case,
        "passed": result.passed,
        "skipped": result.skipped,
        "criterion": result.criterion,
        "summary": result.summary,
        "notes": list(result.notes),
    }


def run_stage(ctx: RunContext) -> Mapping[str, Any] | None:
    """Run all registered validation cases and write ``backtests.json``."""
    names = list_cases()
    if ctx.dry_run:
        return {"cases": names, "dry_run": True}
    results: list[CaseResult] = []
    for name in names:
        log.info("validation case %s (device=%s)", name, ctx.device)
        results.append(run_case(name))
    out_dir = artifact_dir(ctx)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "backtests.json"
    payload = {
        "run_id": ctx.run_id,
        "as_of": ctx.as_of.isoformat(),
        "cpu_only": ctx.cpu_only,
        "cases": [_case_dict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "path": str(out),
        "passed": sum(r.passed and not r.skipped for r in results),
        "failed": sum(not r.passed and not r.skipped for r in results),
        "skipped": sum(r.skipped for r in results),
        "results": results,
    }


def run_report_stage(ctx: RunContext) -> Mapping[str, Any] | None:
    """Write ``docs/validation.md`` from the backtest results (or rerun the cases)."""
    from kairos.validation.report import FIGURE_DIR, draw_figure, render_markdown

    docs_dir = ctx.root / "docs"
    out = docs_dir / "validation.md"
    if ctx.dry_run:
        return {"path": str(out), "dry_run": True}
    prior = ctx.artifacts.get("backtests", {}).get("results")
    results: list[CaseResult] = (
        list(prior) if prior is not None else [run_case(n) for n in list_cases()]
    )
    figures: dict[str, str] = {}
    for result in results:
        path = draw_figure(result, docs_dir / FIGURE_DIR)
        if path is not None:
            figures[result.case] = path.relative_to(docs_dir).as_posix()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render_markdown(results, figures, generated=date.today())
    out.write_text(text, encoding="utf-8", newline="\n")
    return {"path": str(out), "figures": figures}
