"""NeuralProphet ``h*`` cross-check (K-6.05).

The procedure is the same as for Prophet: backtest, score, apply the D3 rule.
The only difference is that NeuralProphet reports **quantiles** rather than one
interval. This module provides:

* :class:`QuantileForecaster`, the protocol a quantile model satisfies, and
  :class:`QuantileAdapter`, which turns one into a
  :class:`~kairos.vinge.schema.Forecaster`. The median becomes ``yhat``, the
  quantile pair matching the interval becomes the band, and every quantile is
  kept as ``q_<p>`` so CRPS uses the quantile form;
* :func:`from_neuralprophet_columns`, which renames NeuralProphet's
  ``yhat1``, ``"yhat1 5.0%"``, ... columns to ``q_<p>``;
* :func:`cross_check`, which compares two ``h*(t)`` tables.

The adapter is model-agnostic and runs on the CPU in tests. Fitting the real
NeuralProphet ensemble on the GPU is the job of Phase 4 (K-4.05).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kairos.types import SeriesFrame
from kairos.vinge.schema import ForecastFrame, quantile_columns

__all__ = [
    "CrossCheck",
    "QuantileAdapter",
    "QuantileForecaster",
    "cross_check",
    "from_neuralprophet_columns",
]

_NP_QUANTILE = re.compile(r"^yhat(\d+) (\d+(?:\.\d+)?)%$")


@runtime_checkable
class QuantileForecaster(Protocol):
    """A model that forecasts quantiles (e.g. a NeuralProphet ensemble).

    ``fit_predict_quantiles`` returns ``ds`` plus one ``q_<p>`` column per
    level in raw positive units, with exactly ``horizon`` rows after the last
    ``history.ds``. Levels must include 0.5 and the pair for the interval.
    """

    def fit_predict_quantiles(self, history: SeriesFrame, horizon: int) -> pd.DataFrame:
        """Fit on ``history`` and return quantile forecasts."""
        ...


@dataclass(frozen=True, slots=True)
class QuantileAdapter:
    """Expose a :class:`QuantileForecaster` as a ``Forecaster``.

    Attributes:
        model: Quantile model (must be picklable).
        interval_width: Central coverage used for ``yhat_lo``/``yhat_hi``.
    """

    model: QuantileForecaster
    interval_width: float = 0.90

    def fit_predict(self, history: SeriesFrame, horizon: int) -> ForecastFrame:
        """Fit and forecast, mapping quantiles to the ``ForecastFrame`` columns."""
        out = self.model.fit_predict_quantiles(history, horizon).copy()
        levels = quantile_columns(out)
        lo_p = round(0.5 - self.interval_width / 2, 6)
        hi_p = round(0.5 + self.interval_width / 2, 6)

        def pick(p: float) -> str:
            for level, col in levels.items():
                if np.isclose(level, p):
                    return col
            raise ValueError(f"quantile {p} missing; have {sorted(levels)}")

        # Monotone rearrangement (Chernozhukov et al. 2010) fixes quantile crossing.
        cols = list(levels.values())
        out[cols] = np.sort(out[cols].to_numpy("float64"), axis=1)
        out["yhat"] = out[pick(0.5)]
        out["yhat_lo"] = out[pick(lo_p)]
        out["yhat_hi"] = out[pick(hi_p)]
        return out[["ds", "yhat", "yhat_lo", "yhat_hi", *cols]]


def from_neuralprophet_columns(
    df: pd.DataFrame, *, step: int = 1, log_space: bool = True
) -> pd.DataFrame:
    """Rename NeuralProphet's quantile columns for one forecast step to ``q_<p>``.

    Args:
        df: NeuralProphet ``predict`` output with ``ds``, ``yhat<step>`` and
            ``"yhat<step> <pct>%"`` columns.
        step: Which ``yhat<step>`` family to take.
        log_space: The model was fitted on ``ln y``; exponentiate.

    Returns:
        ``ds`` plus ``q_<p>`` columns (the median from ``yhat<step>``).
    """
    out = pd.DataFrame({"ds": pd.to_datetime(df["ds"])})
    out["q_0.5"] = df[f"yhat{step}"].to_numpy("float64")
    for col in df.columns:
        match = _NP_QUANTILE.match(str(col))
        if match and int(match.group(1)) == step:
            level = float(match.group(2)) / 100
            out[f"q_{level:g}"] = df[col].to_numpy("float64")
    if log_space:
        qcols = [c for c in out.columns if c != "ds"]
        out[qcols] = np.exp(out[qcols])
    return out


@dataclass(frozen=True, slots=True)
class CrossCheck:
    """Agreement between two ``h*(t)`` tables on their common ``(series, as_of)`` rows.

    Attributes:
        n: Number of common readings.
        spearman: Rank correlation of ``h_star`` (NaN if fewer than 3 readings).
        mean_abs_diff: Mean absolute difference in months.
        share_within: Share of readings within ``tolerance`` months.
        tolerance: Months used for ``share_within``.
    """

    n: int
    spearman: float
    mean_abs_diff: float
    share_within: float
    tolerance: int

    def agrees(self, min_share: float = 0.8) -> bool:
        """Whether at least ``min_share`` of readings are within tolerance."""
        return self.n > 0 and self.share_within >= min_share


def cross_check(primary: pd.DataFrame, secondary: pd.DataFrame, tolerance: int = 3) -> CrossCheck:
    """Compare ``h*(t)`` from two model families (e.g. Prophet vs NeuralProphet)."""
    keys = ["series_id", "as_of"]
    both = primary[[*keys, "h_star"]].merge(
        secondary[[*keys, "h_star"]], on=keys, suffixes=("_a", "_b")
    )
    n = len(both)
    if n == 0:
        return CrossCheck(0, float("nan"), float("nan"), 0.0, tolerance)
    a = both["h_star_a"].to_numpy("float64")
    b = both["h_star_b"].to_numpy("float64")
    diff = np.abs(a - b)
    rho = float(spearmanr(a, b).statistic) if n >= 3 else float("nan")
    return CrossCheck(n, rho, float(diff.mean()), float((diff <= tolerance).mean()), tolerance)
