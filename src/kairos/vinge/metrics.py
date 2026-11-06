"""Log-unit error metrics for the Vinge Horizon (K-6.02, design note D2).

MAPE on ``ln y`` is meaningless: ``ln FLOP`` sits near 57, so a "25 %" error
is a factor of about ``e^14``. Every metric here works on the log scale
directly and has a unit you can state as a factor:

* ``ale`` - absolute log error ``|ln yhat - ln y|`` (in e-folds);
* ``factor_error`` - ``exp(ale)``; ``2.0`` means "off by 2x";
* ``crps`` - continuous ranked probability score of the forecast distribution
  of ``ln y`` (e-folds). With ``q_<p>`` columns it is the quantile (pinball)
  approximation; otherwise a Gaussian is fitted to the median and the central
  band;
* ``band_efolds`` - width of the central band, ``ln yhat_hi - ln yhat_lo``.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm

from kairos.vinge.schema import (
    DEFAULT_INTERVAL,
    BacktestFrame,
    quantile_columns,
    validate_backtest_frame,
)

__all__ = [
    "METRICS",
    "Metric",
    "crps_gaussian",
    "crps_quantiles",
    "error_curve",
    "score_backtest",
]

Metric = Literal["ale", "factor_error", "crps", "band_efolds"]
METRICS: Final[tuple[Metric, ...]] = ("ale", "factor_error", "crps", "band_efolds")
FloatArray = npt.NDArray[np.float64]

_INV_SQRT_PI: Final = 1.0 / np.sqrt(np.pi)


def crps_gaussian(mu: FloatArray, sigma: FloatArray, x: FloatArray) -> FloatArray:
    """Closed-form CRPS of ``N(mu, sigma^2)`` at observation ``x`` (Gneiting et al. 2005)."""
    sigma = np.maximum(np.asarray(sigma, dtype="float64"), 1e-12)
    z = (np.asarray(x, dtype="float64") - mu) / sigma
    out: FloatArray = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - _INV_SQRT_PI)
    return out


def crps_quantiles(levels: FloatArray, q: FloatArray, x: FloatArray) -> FloatArray:
    """CRPS approximated by twice the mean pinball loss over quantile levels.

    Args:
        levels: Shape ``(k,)`` quantile levels in (0, 1).
        q: Shape ``(n, k)`` predicted quantiles.
        x: Shape ``(n,)`` observations.
    """
    levels = np.asarray(levels, dtype="float64")
    diff = np.asarray(x, dtype="float64")[:, None] - np.asarray(q, dtype="float64")
    pinball = np.maximum(levels * diff, (levels - 1) * diff)
    out: FloatArray = 2.0 * pinball.mean(axis=1)
    return out


def score_backtest(bt: BacktestFrame, interval_width: float = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Add log-unit error columns to a ``BacktestFrame``.

    Args:
        bt: Backtest rows (raw units).
        interval_width: Nominal coverage of ``yhat_lo``/``yhat_hi``, used for the
            Gaussian CRPS when there are no ``q_<p>`` columns.

    Returns:
        A copy of ``bt`` with ``log_error`` (signed, ``ln yhat - ln y``) and the
        columns in :data:`METRICS`, plus ``covered`` (``y`` inside the band).
    """
    if not 0.0 < interval_width < 1.0:
        raise ValueError("interval_width must be in (0, 1)")
    validate_backtest_frame(bt)
    out = bt.copy()
    log_y = np.log(out["y"].to_numpy("float64"))
    log_hat = np.log(out["yhat"].to_numpy("float64"))
    log_lo = np.log(out["yhat_lo"].to_numpy("float64"))
    log_hi = np.log(out["yhat_hi"].to_numpy("float64"))
    out["log_error"] = log_hat - log_y
    out["ale"] = np.abs(out["log_error"])
    out["factor_error"] = np.exp(out["ale"])
    out["band_efolds"] = log_hi - log_lo
    out["covered"] = (out["y"] >= out["yhat_lo"]) & (out["y"] <= out["yhat_hi"])
    qcols = quantile_columns(out)
    if len(qcols) >= 3:
        levels = np.fromiter(qcols.keys(), dtype="float64")
        q = np.log(out[list(qcols.values())].to_numpy("float64"))
        out["crps"] = crps_quantiles(levels, np.sort(q, axis=1), log_y)
    else:
        z = float(norm.ppf(0.5 + interval_width / 2))
        sigma = (log_hi - log_lo) / (2 * z)
        out["crps"] = crps_gaussian(log_hat, sigma, log_y)
    return out


def error_curve(
    scored: pd.DataFrame,
    metric: Metric = "ale",
    *,
    agg: Literal["median", "mean"] = "median",
    by: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Aggregate a scored backtest into one error value per horizon.

    Args:
        scored: Output of :func:`score_backtest`.
        metric: Column to aggregate. ``factor_error`` is aggregated as
            ``exp(agg(ale))`` so mean and median stay on the log scale.
        agg: ``"median"`` (robust, default) or ``"mean"``.
        by: Extra grouping keys, e.g. ``("series_id",)``.

    Returns:
        Columns ``*by, horizon, error, n``, sorted by horizon.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    col = "ale" if metric == "factor_error" else metric
    keys = [*by, "horizon"]
    grouped = scored.groupby(keys, sort=True)[col]
    out = grouped.agg(agg).rename("error").to_frame()
    out["n"] = grouped.size()
    out = out.reset_index()
    if metric == "factor_error":
        out["error"] = np.exp(out["error"])
    return out
