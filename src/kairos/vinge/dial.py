"""The Vinge dial: current ``h*``, its trend and a state (K-6.07).

The dial reads an ``h*(t)`` table (see :mod:`kairos.vinge.series`) and returns
a :class:`kairos.types.DialReading` with ``dial="vinge"``:

* ``value``: median over series of the latest ``h*`` (months);
* ``ci_lo``/``ci_hi``: 10th and 90th percentile of the latest ``h*`` across
  series (equal to ``value`` for a single series);
* ``level``: the state below; ``detail`` carries each series' reading and
  trend test, plus the thresholds used.

States, with every threshold loaded from the published ``thresholds.yaml``:

======== ===== ==============================================================
state    level condition
======== ===== ==============================================================
normal   0     fewer than ``min_shrinking_series`` series shrink significantly
watch    1     at least ``min_shrinking_series`` series shrink
warning  2     ``confirm_shrinking_series`` shrink and median ``h*`` <= ``warning_months``
critical 3     as warning, with median ``h*`` <= ``critical_months``
======== ===== ==============================================================

Level 4 (Kairos) is reserved for the governor, which combines all four dials.
"""

from __future__ import annotations

from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kairos.types import DialReading
from kairos.vinge.series import HorizonSpec
from kairos.vinge.trend import shrinkage_by_series

__all__ = [
    "STATES",
    "VingeState",
    "VingeThresholds",
    "load_thresholds",
    "spec_from_thresholds",
    "vinge_dial",
]

VingeState = Literal["normal", "watch", "warning", "critical"]
STATES: Final[tuple[VingeState, ...]] = ("normal", "watch", "warning", "critical")


class VingeThresholds(BaseModel):
    """Published parameters of the Vinge dial (see ``thresholds.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    tolerance_factor: float = Field(gt=1.0)
    metric: Literal["ale", "crps"]
    k_efolds: float = Field(gt=0)
    window_months: int = Field(ge=3)
    alpha: float = Field(gt=0, lt=1)
    min_shrinking_series: int = Field(ge=1)
    confirm_shrinking_series: int = Field(ge=1)
    warning_months: float = Field(gt=0)
    critical_months: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> VingeThresholds:
        if self.critical_months > self.warning_months:
            raise ValueError("critical_months must be <= warning_months")
        if self.confirm_shrinking_series < self.min_shrinking_series:
            raise ValueError("confirm_shrinking_series must be >= min_shrinking_series")
        return self


def load_thresholds(path: Path | str | None = None) -> VingeThresholds:
    """Load dial thresholds from ``path``, or the packaged ``thresholds.yaml``."""
    if path is None:
        text = files("kairos.vinge").joinpath("thresholds.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return VingeThresholds.model_validate(yaml.safe_load(text))


def spec_from_thresholds(th: VingeThresholds) -> HorizonSpec:
    """The :class:`~kairos.vinge.series.HorizonSpec` matching published thresholds."""
    return HorizonSpec(
        tolerance_factor=th.tolerance_factor,
        metric=th.metric,
        k_efolds=th.k_efolds,
        window=th.window_months,
    )


def _state(n_shrinking: int, current: float, th: VingeThresholds) -> VingeState:
    if n_shrinking >= th.confirm_shrinking_series:
        if current <= th.critical_months:
            return "critical"
        if current <= th.warning_months:
            return "warning"
    if n_shrinking >= th.min_shrinking_series:
        return "watch"
    return "normal"


def vinge_dial(
    hstar: pd.DataFrame,
    thresholds: VingeThresholds | None = None,
    *,
    as_of: date | None = None,
) -> DialReading:
    """Build the Vinge dial reading from an ``h*(t)`` table.

    Args:
        hstar: Output of :func:`kairos.vinge.series.hstar_series`.
        thresholds: Dial parameters; default :func:`load_thresholds`.
        as_of: Vintage date of the reading; default the latest ``as_of`` in ``hstar``.
            Readings after it are ignored.

    Raises:
        ValueError: If no readings are available on or before ``as_of``.
    """
    th = thresholds or load_thresholds()
    df = hstar.copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    cut = pd.Timestamp(as_of) if as_of is not None else df["as_of"].max()
    df = df.loc[df["as_of"] <= cut]
    if df.empty:
        raise ValueError("no h*(t) readings on or before as_of")
    latest = df.sort_values("as_of").groupby("series_id").tail(1).set_index("series_id")
    values = latest["h_star"].to_numpy("float64")
    current = float(np.median(values))
    lo, hi = (float(v) for v in np.percentile(values, [10, 90]))

    trends = shrinkage_by_series(df, alpha=th.alpha)
    trend_by_id = trends.set_index("series_id") if not trends.empty else pd.DataFrame()
    n_shrinking = int(trends["shrinking"].sum()) if not trends.empty else 0
    median_slope = float(trends["slope"].median()) if not trends.empty else float("nan")
    state = _state(n_shrinking, current, th)

    per_series: dict[str, dict[str, object]] = {}
    for sid, row in latest.iterrows():
        entry: dict[str, object] = {
            "h_star": float(row["h_star"]),
            "censored": bool(row.get("censored", False)),
            "as_of": pd.Timestamp(row["as_of"]).date().isoformat(),
        }
        if sid in trend_by_id.index:
            t = trend_by_id.loc[sid]
            entry |= {
                "slope_months_per_year": float(t["slope"]),
                "slope_ci": [float(t["slope_lo"]), float(t["slope_hi"])],
                "p_value": float(t["p_value"]),
                "shrinking": bool(t["shrinking"]),
            }
        per_series[str(sid)] = entry

    return DialReading(
        dial="vinge",
        value=current,
        level=STATES.index(state),
        ci_lo=min(lo, current),
        ci_hi=max(hi, current),
        as_of=pd.Timestamp(cut).date(),
        detail={
            "state": state,
            "unit": "months",
            "n_series": len(latest),
            "n_shrinking": n_shrinking,
            "median_slope_months_per_year": median_slope,
            "any_censored": bool(latest.get("censored", pd.Series(dtype=bool)).any()),
            "series": per_series,
            "thresholds": th.model_dump(),
        },
    )
