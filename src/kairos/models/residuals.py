"""One-sided residual tests: does reality keep beating the boring-world forecast?

Residuals are taken on the log scale against a forecast made **before** the
observations (out of sample)::

    r_t = ln y_t - ln yhat_t,        z_t = r_t / sigma_t

with ``sigma_t`` recovered from the forecast interval:
``sigma_t = (ln upper_t - ln lower_t) / (2 * Phi^-1((1 + w) / 2))`` for interval width ``w``.

Two tests, both one-sided ("reality > forecast"):

* **Sign test.** Under the null, each residual is positive with probability 1/2;
  ``p = P(Binomial(n, 1/2) >= n_positive)``. Distribution-free but ignores size.
* **Upper CUSUM** (Page 1954). ``S_t = max(0, S_{t-1} + z_t - k)``; alarm when
  ``S_t > h``. Defaults ``k = 0.5``, ``h = 5`` are the textbook choice for detecting
  a one-sigma upward shift, with an in-control average run length near 900.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from kairos.models.null_model import NullModelFit

__all__ = ["ResidualTest", "cusum_upper", "one_sided_residual_test", "residuals_vs_fit"]


def cusum_upper(
    z: np.ndarray[Any, Any], *, k: float = 0.5
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Upper CUSUM path ``S_t = max(0, S_{t-1} + z_t - k)`` with ``S_0 = 0``."""
    out = np.empty(len(z), dtype="float64")
    s = 0.0
    for i, value in enumerate(np.asarray(z, dtype="float64")):
        s = max(0.0, s + value - k)
        out[i] = s
    return out


@dataclass(frozen=True)
class ResidualTest:
    """Result of :func:`one_sided_residual_test`.

    Attributes:
        residuals: Frame ``ds, r, z, cusum`` (log residual, standardised, CUSUM path).
        n: Observations tested.
        n_positive: Residuals above zero (reality beat the median forecast).
        sign_p: One-sided sign-test p-value.
        cusum_alarm: First date the CUSUM crossed ``h``, or ``None``.
        k: CUSUM reference value.
        h: CUSUM decision threshold.
    """

    residuals: pd.DataFrame
    n: int
    n_positive: int
    sign_p: float
    cusum_alarm: pd.Timestamp | None
    k: float
    h: float

    @property
    def alarm(self) -> bool:
        """The CUSUM crossed its threshold."""
        return self.cusum_alarm is not None

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "n": self.n,
            "n_positive": self.n_positive,
            "sign_p": self.sign_p,
            "cusum_max": float(self.residuals["cusum"].max()) if self.n else 0.0,
            "cusum_alarm": None if self.cusum_alarm is None else self.cusum_alarm.strftime("%Y-%m"),
            "k": self.k,
            "h": self.h,
        }


def one_sided_residual_test(
    observed: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    interval_width: float = 0.9,
    k: float = 0.5,
    h: float = 5.0,
) -> ResidualTest:
    """Sign test and upper CUSUM on log residuals of ``observed`` against ``forecast``.

    Args:
        observed: ``ds`` and raw positive ``y``.
        forecast: Raw-unit forecast with ``ds, yhat, yhat_lower, yhat_upper``.
        interval_width: Width of the forecast interval (to recover ``sigma``).
        k: CUSUM reference value in sigma units.
        h: CUSUM threshold in sigma units.
    """
    if not 0 < interval_width < 1:
        raise ValueError("interval_width must be in (0, 1)")
    obs = observed[["ds", "y"]].dropna()
    fc = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    merged = obs.merge(fc, on="ds", how="inner").sort_values("ds", ignore_index=True)
    r = np.log(merged["y"].to_numpy(dtype="float64")) - np.log(merged["yhat"].to_numpy())
    z_q = float(stats.norm.ppf((1 + interval_width) / 2))
    width = np.log(merged["yhat_upper"].to_numpy()) - np.log(merged["yhat_lower"].to_numpy())
    sigma = np.maximum(width / (2 * z_q), 1e-12)
    z = r / sigma
    path = cusum_upper(z, k=k)
    residuals = pd.DataFrame({"ds": merged["ds"], "r": r, "z": z, "cusum": path})

    n = len(r)
    n_pos = int((r > 0).sum())
    sign_p = float(stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue) if n else 1.0
    crossed = np.flatnonzero(path > h)
    alarm = pd.Timestamp(residuals["ds"].iloc[crossed[0]]) if crossed.size else None
    return ResidualTest(residuals, n, n_pos, sign_p, alarm, k, h)


def residuals_vs_fit(
    fit: NullModelFit,
    observed: pd.DataFrame,
    *,
    k: float = 0.5,
    h: float = 5.0,
) -> ResidualTest:
    """Test observations **after** the fit's history against its forecast.

    Args:
        fit: A null model fitted on data up to its cutoff.
        observed: Later observations (``ds``, raw ``y``); in-sample rows are ignored.
        k: CUSUM reference value.
        h: CUSUM threshold.
    """
    cutoff = fit.history["ds"].max()
    later = observed.loc[pd.to_datetime(observed["ds"]) > cutoff]
    width = float(fit.params.get("interval_width", 0.9))
    return one_sided_residual_test(later, fit.forecast, interval_width=width, k=k, h=h)
