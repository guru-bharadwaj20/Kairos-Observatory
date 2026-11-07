"""The acceleration dial (K-5.14).

Combines the Phase 5 evidence into one :class:`kairos.types.DialReading`:

* per-metric ``δ`` with block-bootstrap intervals (K-5.04/K-5.05),
* Benjamini–Hochberg control of "``δ > 0``" across metrics (K-5.12),
* confirmation across independent source groups (K-5.13),
* the pooled ``t_c`` interval and whether it is **narrowing** relative to the
  previous reading,
* the LPPLS confidence indicator (K-5.11), if supplied.

``value`` is the median ``δ`` of the significant metrics (of all metrics when none is
significant); ``ci_lo``/``ci_hi`` are the medians of their interval bounds. The
state ladder and its ``level``:

===== ============== =========================================================
level state          condition
===== ============== =========================================================
0     ``normal``     no metric has a significant ``δ > 0``
1     ``watch``      at least one significant metric
2     ``confirmed``  significant metrics span ≥ ``min_groups`` source groups
3     ``converging`` confirmed and (``t_c`` interval narrowing or LPPLS high)
4     ``kairos``     confirmed, narrowing and LPPLS confidence high
===== ============== =========================================================

Thresholds live in :class:`AccelerationDialConfig`; the published tripwire YAML of
Phase 8 is expected to construct it, so no threshold is hard-wired at call sites.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from kairos.accel.bootstrap import BootstrapResult
from kairos.accel.independence import confirm_by_source_group
from kairos.accel.multitest import delta_significance
from kairos.types import DialReading, SeriesFrame

__all__ = ["STATES", "AccelerationDialConfig", "acceleration_dial"]

STATES = ("normal", "watch", "confirmed", "converging", "kairos")


class AccelerationDialConfig(BaseModel):
    """Parameters of the acceleration dial.

    Attributes:
        q: False discovery rate for "``δ > 0``".
        level: Level of the ``δ`` and ``t_c`` intervals.
        min_groups: Independent source groups needed for confirmation.
        lppls_high: LPPLS confidence at or above which the signature counts as high.
        narrowing_ratio: The ``t_c`` interval counts as narrowing when its width is
            below this fraction of the previous reading's width.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    q: float = Field(default=0.1, gt=0, lt=1)
    level: float = Field(default=0.9, gt=0, lt=1)
    min_groups: int = Field(default=3, ge=1)
    lppls_high: float = Field(default=0.5, ge=0, le=1)
    narrowing_ratio: float = Field(default=0.9, gt=0, le=1)


def _median(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.median(finite)) if finite else float("nan")


def _json_float(x: float) -> float | None:
    return x if math.isfinite(x) else None


def acceleration_dial(
    results: Mapping[str, BootstrapResult],
    groups: Mapping[str, str] | SeriesFrame,
    *,
    as_of: date,
    lppls_confidence: float | None = None,
    previous: DialReading | None = None,
    config: AccelerationDialConfig | None = None,
) -> DialReading:
    """Build the acceleration dial reading.

    Args:
        results: Series ID → bootstrap result (one per metric).
        groups: ``series_id → source_group`` mapping or the ``SeriesFrame``.
        as_of: Data vintage of the reading.
        lppls_confidence: Latest LPPLS confidence (0–1), if computed.
        previous: Previous acceleration reading, for the narrowing check.
        config: Dial parameters; defaults to :class:`AccelerationDialConfig`.

    Raises:
        ValueError: If ``results`` is empty.
    """
    cfg = config or AccelerationDialConfig()
    if not results:
        raise ValueError("need at least one metric")
    table = delta_significance(results, q=cfg.q, level=cfg.level)
    conf = confirm_by_source_group(table, groups, min_groups=cfg.min_groups)
    sig = table[table["significant"]]
    use = sig if len(sig) else table

    value = _median(use["delta"].tolist())
    ci_lo = _median(use["delta_lo"].tolist())
    ci_hi = _median(use["delta_hi"].tolist())
    if math.isnan(value):
        value, ci_lo, ci_hi = 0.0, 0.0, 0.0

    tc_bounds = [results[s].interval(cfg.level, "t_c") for s in sig["series_id"]]
    tc_lo = _median([b[0] for b in tc_bounds]) if tc_bounds else float("nan")
    tc_hi = _median([b[1] for b in tc_bounds]) if tc_bounds else float("nan")
    width = tc_hi - tc_lo if tc_bounds else float("nan")

    prev_width = float("nan")
    if previous is not None:
        pw = previous.detail.get("t_c_width")
        prev_width = float(pw) if pw is not None else float("nan")
    ratio = width / prev_width if math.isfinite(width) and math.isfinite(prev_width) else math.nan
    narrowing = bool(math.isfinite(ratio) and ratio < cfg.narrowing_ratio)
    lppls_high = lppls_confidence is not None and lppls_confidence >= cfg.lppls_high

    if conf.confirmed and narrowing and lppls_high:
        level = 4
    elif conf.confirmed and (narrowing or lppls_high):
        level = 3
    elif conf.confirmed:
        level = 2
    elif len(sig):
        level = 1
    else:
        level = 0

    detail: dict[str, Any] = {
        "state": STATES[level],
        "metrics": [
            {k: (_json_float(v) if isinstance(v, float) else v) for k, v in row.items()}
            for row in table.to_dict(orient="records")
        ],
        "n_metrics": len(table),
        "n_significant": len(sig),
        "n_groups": conf.n_groups,
        "groups": conf.groups,
        "confirmed": conf.confirmed,
        "t_c_ci": [_json_float(tc_lo), _json_float(tc_hi)],
        "t_c_width": _json_float(width),
        "t_c_narrowing_ratio": _json_float(ratio),
        "t_c_narrowing": narrowing,
        "lppls_confidence": lppls_confidence,
        "config": cfg.model_dump(),
    }
    return DialReading(
        dial="acceleration",
        value=value,
        level=level,
        ci_lo=min(ci_lo, ci_hi),
        ci_hi=max(ci_lo, ci_hi),
        as_of=as_of,
        detail=detail,
    )
