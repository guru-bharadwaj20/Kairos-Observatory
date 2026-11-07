"""Public scorecard: running calibration and CRPS from the ledger (K-10.07).

Everything here is computed from ``score`` entries alone, so anyone holding the
public ledger can regenerate the table and check it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd

from kairos.ledger.chain import Ledger, LedgerEntry
from kairos.ledger.scoring import ForecastScore

__all__ = ["PIT_BINS", "load_scores", "scorecard", "scorecard_markdown"]

PIT_BINS = 10
COLUMNS = [
    "series_id",
    "model",
    "n_forecasts",
    "n_points",
    "n_counterfactual",
    "mean_crps",
    "coverage",
    "nominal",
    "pit_ks",
    "claims_within",
    "pit_hist",
]


def load_scores(entries: Sequence[LedgerEntry]) -> list[ForecastScore]:
    """Return the scores in ``entries``, dropping any superseded by a correction."""
    superseded = {e.payload["supersedes"] for e in entries if e.kind == "correction"}
    return [
        ForecastScore.model_validate(e.payload)
        for e in entries
        if e.kind == "score" and e.entry_hash not in superseded
    ]


def _pit_ks(pits: npt.NDArray[np.float64]) -> float:
    """Kolmogorov-Smirnov distance of PIT values from Uniform(0, 1)."""
    if pits.size == 0:
        return float("nan")
    x = np.sort(pits)
    n = x.size
    upper = np.arange(1, n + 1) / n - x
    lower = x - np.arange(n) / n
    return float(max(upper.max(), lower.max()))


def _row(series_id: str, model: str, scores: list[ForecastScore]) -> dict[str, object]:
    points = [p for s in scores for p in s.points]
    pits = np.array([p.pit for p in points], dtype=float)
    covered = [p.covered for p in points if p.covered is not None]
    claims = [c.within_interval for s in scores for c in s.claims]
    hist, _ = np.histogram(pits, bins=PIT_BINS, range=(0.0, 1.0))
    nominal = {s.coverage_level for s in scores}
    return {
        "series_id": series_id,
        "model": model,
        "n_forecasts": len(scores),
        "n_points": len(points),
        "n_counterfactual": sum(s.counterfactual for s in scores),
        "mean_crps": float(np.mean([p.crps for p in points])) if points else float("nan"),
        "coverage": float(np.mean(covered)) if covered else float("nan"),
        "nominal": nominal.pop() if len(nominal) == 1 else float("nan"),
        "pit_ks": _pit_ks(pits),
        "claims_within": float(np.mean(claims)) if claims else float("nan"),
        "pit_hist": " ".join(str(int(c)) for c in hist),
    }


def scorecard(source: Ledger | Sequence[LedgerEntry]) -> pd.DataFrame:
    """Return one row per (series, model) plus an ``ALL`` row.

    Columns: ``n_forecasts``, ``n_points``, ``n_counterfactual``, ``mean_crps``
    (raw units, so comparable only within a series), ``coverage`` (empirical) next
    to ``nominal``, ``pit_ks`` (0 = perfectly calibrated), ``claims_within``
    (fraction of conditional claims resolved inside their interval) and
    ``pit_hist`` (PIT counts in 10 equal bins; flat means calibrated).
    """
    entries = source.entries() if isinstance(source, Ledger) else list(source)
    scores = load_scores(entries)
    groups: dict[tuple[str, str], list[ForecastScore]] = {}
    for s in scores:
        groups.setdefault((s.series_id, s.model), []).append(s)
    rows = [_row(sid, model, group) for (sid, model), group in sorted(groups.items())]
    if scores:
        all_row = _row("ALL", "ALL", scores)
        all_row["mean_crps"] = float("nan")  # CRPS is in each series' own units
        rows.append(all_row)
    return pd.DataFrame(rows, columns=COLUMNS)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "-" if np.isnan(value) else f"{value:.3g}"
    return str(value)


def scorecard_markdown(table: pd.DataFrame, *, generated: date | None = None) -> str:
    """Render :func:`scorecard` output as a Markdown page."""
    lines = ["# KAIROS forecast scorecard", ""]
    if generated is not None:
        lines += [f"Generated {generated.isoformat()} from `ledger/forecasts.jsonl`.", ""]
    if table.empty:
        return "\n".join([*lines, "No forecasts have matured yet.", ""])
    lines += [
        "Coverage is the share of observations inside the central interval; compare it",
        "with `nominal`. `pit_ks` is the KS distance of PIT values from uniform.",
        "Counterfactual forecasts were overtaken by interventions logged later.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for record in table.to_dict(orient="records"):
        lines.append("| " + " | ".join(_fmt(record[c]) for c in COLUMNS) + " |")
    return "\n".join([*lines, ""])
