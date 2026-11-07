"""Out-of-sample evaluation: hold-out splits, skill per horizon and calibration.

NeuralProphet's ``predict`` puts the ``h``-step-ahead forecast *of* month ``t``
in column ``yhat<h>`` on row ``t``. Every metric here is computed on rows whose
month lies in the hold-out window, so models with different lag windows are
scored on exactly the same targets (CONTRIBUTING D9: out-of-sample only).
Errors are in the model's units: log units for growth series, so an absolute
error of ``0.69`` is a factor of two.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_QUANTILES",
    "Holdout",
    "calibration_table",
    "holdout_split",
    "horizon_errors",
    "quantile_column",
]

DEFAULT_QUANTILES: tuple[float, float, float] = (0.05, 0.5, 0.95)
"""Quantiles forecast by default: a central 90 % interval around the median."""


@dataclass(frozen=True)
class Holdout:
    """A train / test split in time.

    Attributes:
        train: Rows used for fitting.
        full: Training rows plus the test rows (predict on this: lags need history).
        test_start: First month of the hold-out window.
    """

    train: pd.DataFrame
    full: pd.DataFrame
    test_start: pd.Timestamp


def holdout_split(df: pd.DataFrame, test_months: int) -> Holdout:
    """Hold out the last ``test_months`` months of ``df``.

    Raises:
        ValueError: If ``test_months`` is not in ``[1, len(df) - 1]``.
    """
    if not 1 <= test_months < len(df):
        raise ValueError(f"test_months must be in [1, {len(df) - 1}], got {test_months}")
    full = df.sort_values("ds").reset_index(drop=True)
    cut = len(full) - test_months
    return Holdout(full.iloc[:cut].copy(), full, pd.Timestamp(full["ds"].iloc[cut]))


def quantile_column(horizon: int, q: float) -> str:
    """NeuralProphet column name of quantile ``q`` at ``horizon`` (``"yhat1 5.0%"``)."""
    return f"yhat{horizon} {round(q * 100, 1)}%"


def _test_rows(forecast: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    return forecast.loc[forecast["ds"] >= test_start]


def horizon_errors(
    forecast: pd.DataFrame, n_forecasts: int, test_start: pd.Timestamp
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Mean absolute error of the point forecast at each horizon ``1..n_forecasts``.

    Raises:
        ValueError: If some horizon has no scored row.
    """
    rows = _test_rows(forecast, test_start)
    y = rows["y"].astype(np.float64)
    out = np.empty(n_forecasts)
    for h in range(1, n_forecasts + 1):
        err = (y - rows[f"yhat{h}"].astype(np.float64)).abs().dropna()
        if err.empty:
            raise ValueError(f"no out-of-sample rows for horizon {h}")
        out[h - 1] = float(err.mean())
    return out


def calibration_table(
    forecast: pd.DataFrame,
    n_forecasts: int,
    test_start: pd.Timestamp,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Empirical coverage of the central interval on held-out rows, per horizon.

    The interval runs from the lowest to the highest quantile in ``quantiles``.

    Returns:
        One row per horizon: ``horizon``, ``nominal`` coverage, ``coverage``
        (share of held-out ``y`` inside the interval), ``below`` / ``above``
        (shares outside each side), ``mean_width`` and ``n``.
    """
    qs = sorted(q for q in quantiles if q != 0.5)
    if len(qs) < 2:
        raise ValueError("need a lower and an upper quantile")
    lo_q, hi_q = qs[0], qs[-1]
    rows = _test_rows(forecast, test_start)
    records = []
    for h in range(1, n_forecasts + 1):
        part = rows[["y", quantile_column(h, lo_q), quantile_column(h, hi_q)]]
        part = part.astype(np.float64).dropna()
        y, lo, hi = (part.iloc[:, i].to_numpy() for i in range(3))
        n = len(y)
        records.append(
            {
                "horizon": h,
                "nominal": hi_q - lo_q,
                "coverage": float(np.mean((y >= lo) & (y <= hi))) if n else np.nan,
                "below": float(np.mean(y < lo)) if n else np.nan,
                "above": float(np.mean(y > hi)) if n else np.nan,
                "mean_width": float(np.mean(hi - lo)) if n else np.nan,
                "n": n,
            }
        )
    return pd.DataFrame.from_records(records)
