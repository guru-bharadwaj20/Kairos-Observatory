"""K-9.09: build ``docs/validation.md`` from the registered cases.

Every case is run, its figure drawn, and the outcome written down whether it
passed, failed or was skipped. Failures and caveats get their own section: the
report exists to show where the detectors are wrong.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from kairos.validation.cases import CaseResult, list_cases, run_case

if TYPE_CHECKING:
    from kairos.commands.run import RunContext

__all__ = ["draw_figure", "render_markdown", "run_stage", "write_report"]

FIGURE_DIR = Path("figures") / "validation"


def _status(result: CaseResult) -> str:
    return "skipped" if result.skipped else ("pass" if result.passed else "FAIL")


def draw_figure(result: CaseResult, out_dir: Path) -> Path | None:
    """Draw one PNG per case with a table; returns its path (``None`` if no table)."""
    if result.table is None or result.table.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = result.table
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=110)
    if "conditional" in table.columns:
        ax.semilogy(table["ds"], table["actual"], "ko", label="reported")
        ax.semilogy(table["ds"], table["conditional"], label="with lockdown event")
        ax.semilogy(table["ds"], table["counterfactual"], "--", label="no-event counterfactual")
        ax.set_ylabel("weekly new cases")
    elif "t_c" in table.columns:
        x = pd.to_datetime(table["as_of"])
        if "delta" in table.columns and table["t_c"].isna().mean() > 0.3:
            ax.plot(x, table["delta"], "o-", label="δ")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_ylabel("δ")
        else:
            ax.plot(x, table["t_c"], "o-", label="t_c")
            if {"t_c_lo", "t_c_hi"} <= set(table.columns):
                ax.fill_between(x, table["t_c_lo"], table["t_c_hi"], alpha=0.2, label="interval")
            ax.set_ylabel("t_c (decimal year)")
        alarms = table[table["alarm"].astype(bool)]
        if len(alarms):
            ax.plot(pd.to_datetime(alarms["as_of"]), [ax.get_ylim()[0]] * len(alarms), "r|",
                    ms=12, label="alarm")  # fmt: skip
        ax.set_xlabel("as-of date")
    else:
        plt.close(fig)
        return None
    ax.set_title(f"{result.case}: {_status(result)}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.case}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) >= 100 and abs(value) < 1e5 else f"{value:.4g}"
    if isinstance(value, tuple | list):
        return "(" + ", ".join(_value(v) for v in value) + ")"
    return str(value)


def render_markdown(
    results: Sequence[CaseResult], figures: dict[str, str], *, generated: date
) -> str:
    """Render the report body."""
    lines = [
        "# Historical validation",
        "",
        f"Generated {generated.isoformat()} by `kairos validate report`. Each case replays "
        "the detectors on a curve whose ending is known, using only data available at "
        "each as-of date (Phase 9, K-9.01 to K-9.08).",
        "",
        "Detectors are the self-contained baselines in `kairos.validation` "
        "(`HyperbolicDetector`, `LPPLSDetector`, `LocalDeltaDetector`) until the Phase 5 "
        "detectors are registered in `kairos.validation.detectors.DETECTORS`.",
        "",
        "## Summary",
        "",
        "| Case | Result | Criterion |",
        "|---|:-:|---|",
    ]
    lines += [f"| {r.case} | {_status(r)} | {r.criterion} |" for r in results]
    for r in results:
        lines += ["", f"## {r.case}", "", f"**Result: {_status(r)}.** {r.criterion}.", ""]
        lines += [f"- `{k}`: {_value(v)}" for k, v in r.summary.items()]
        if r.case in figures:
            lines += ["", f"![{r.case}]({figures[r.case]})"]
    lines += ["", "## Failures and caveats", ""]
    failed = [r for r in results if not r.passed]
    if failed:
        lines += [f"- **{r.case}** ({_status(r)}): " + "; ".join(r.notes or ["no notes"])
                  for r in failed]  # fmt: skip
    else:
        lines.append("- No case failed its stated criterion. That does not make the "
                     "detectors right; the caveats below say where they are weak.")  # fmt: skip
    lines += [
        "- **Known weakness, LPPLS baseline:** it alarms from the first replay month and "
        "keeps alarming after the crash (`post_peak_alarm_rate` = 1). It times bubbles but "
        "has no specificity; the lead times above are therefore upper bounds.",
        "- **Known weakness, population:** the passing specification was chosen after "
        "seeing the data; across reasonable specifications `t_c` spans more than a century.",
        "- **Reference values:** the dot-com window is a coarse reading of Johansen & "
        "Sornette (2000) and should be checked against the paper's tables.",
    ]
    for r in results:
        if r.passed:
            lines += [f"- {r.case}: {note}" for note in r.notes]
    return "\n".join(lines) + "\n"


def write_report(
    docs_dir: Path = Path("docs"), *, cases: Sequence[str] | None = None
) -> tuple[Path, list[CaseResult]]:
    """Run ``cases`` (default: all), draw figures and write ``docs_dir/validation.md``."""
    results = [run_case(name) for name in (cases or list_cases())]
    figures: dict[str, str] = {}
    for result in results:
        path = draw_figure(result, docs_dir / FIGURE_DIR)
        if path is not None:
            figures[result.case] = path.relative_to(docs_dir).as_posix()
    out = docs_dir / "validation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render_markdown(results, figures, generated=date.today())
    out.write_text(text, encoding="utf-8", newline="\n")
    return out, results


def run_stage(ctx: RunContext) -> Mapping[str, Any] | None:
    """``kairos run validation`` report stage; see :mod:`kairos.validation.runner`."""
    from kairos.validation.runner import run_report_stage

    return run_report_stage(ctx)
