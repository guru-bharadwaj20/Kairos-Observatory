"""Data quality checks, run on every refresh.

=============== ======== ==========================================================
check           severity meaning
=============== ======== ==========================================================
``monotonic``   error    ``ds`` is not strictly increasing in file order
``duplicates``  error    repeated ``(series_id, ds)`` rows
``unit``        error    ``unit`` column differs from the metadata or is mixed
``side``        error    ``side`` column differs from the metadata
``gaps``        warning  missing months inside a mostly monthly series
``outliers``    warning  month-on-month jumps beyond ``z_max`` robust z-scores
                         (on the series' model transform)
=============== ======== ==========================================================

Sparse series (fewer than half of the months between first and last point
present) are treated as irregular and skip the gap check.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from kairos.data.store import Dataset
from kairos.data.transforms import transform_values

__all__ = [
    "QualityError",
    "QualityIssue",
    "check_dataset",
    "outlier_flags",
    "quality_report",
    "raise_on_errors",
]

Z_MAX: Final = 6.0
MAD_FLOOR: Final = 1e-3
"""Smallest step scale (transform units), so near-constant growth is not all outliers."""
Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One finding of a quality check."""

    series_id: str
    check: str
    severity: Severity
    detail: str


class QualityError(ValueError):
    """Raised when a refresh finds error-level quality issues."""


def outlier_flags(d: Dataset, z_max: float = Z_MAX) -> pd.Series:
    """Flag points whose step from the previous point is a robust-z outlier.

    The step is taken on the series' transform (``log``/``logit``/``none``); the
    scale is the median absolute deviation of all steps.
    """
    f = d.frame
    if len(f) < 5:
        return pd.Series(False, index=f.index)
    steps = np.diff(transform_values(f["y"].to_numpy(), d.meta.transform))
    med = np.median(steps)
    mad = max(1.4826 * float(np.median(np.abs(steps - med))), MAD_FLOOR)
    z = np.abs(steps - med) / mad
    return pd.Series(np.r_[False, z > z_max], index=f.index)


def check_dataset(d: Dataset, *, z_max: float = Z_MAX) -> list[QualityIssue]:
    """Run every check on one dataset and return the findings."""
    f, sid = d.frame, d.meta.series_id
    out: list[QualityIssue] = []

    def add(check: str, sev: Severity, detail: str) -> None:
        out.append(QualityIssue(sid, check, sev, detail))

    if not f["ds"].is_monotonic_increasing or f["ds"].duplicated().any():
        add("monotonic", "error", "ds is not strictly increasing")
    n_dup = int(f.duplicated(["series_id", "ds"]).sum())
    if n_dup:
        add("duplicates", "error", f"{n_dup} duplicate rows")
    units = set(f["unit"].unique()) if "unit" in f else set()
    if units and units != {d.meta.unit}:
        add("unit", "error", f"units {sorted(units)} != metadata {d.meta.unit!r}")
    if "side" in f and set(f["side"].unique()) != {d.meta.side}:
        add("side", "error", f"side column != metadata {d.meta.side!r}")
    if len(f) >= 2:
        span = pd.date_range(f["ds"].min(), f["ds"].max(), freq="MS")
        missing = len(span) - f["ds"].nunique()
        if missing and f["ds"].nunique() >= 0.5 * len(span):
            add("gaps", "warning", f"{missing} missing months")
    n_out = int(outlier_flags(d, z_max).sum())
    if n_out:
        add("outliers", "warning", f"{n_out} steps beyond {z_max} robust z")
    return out


def quality_report(datasets: Iterable[Dataset], *, z_max: float = Z_MAX) -> pd.DataFrame:
    """Check many datasets; return one row per issue (``series_id``, ``check``, ...)."""
    rows = [asdict(i) for d in datasets for i in check_dataset(d, z_max=z_max)]
    return pd.DataFrame(rows, columns=["series_id", "check", "severity", "detail"])


def raise_on_errors(report: pd.DataFrame) -> None:
    """Raise :class:`QualityError` if the report holds any error-level issue."""
    errors = report[report["severity"] == "error"]
    if not errors.empty:
        lines = "; ".join(f"{r.series_id}: {r.check} ({r.detail})" for r in errors.itertuples())
        raise QualityError(f"data quality errors: {lines}")
