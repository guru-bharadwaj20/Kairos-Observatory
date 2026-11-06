"""Sustained-positive test for ``dG/dt`` (K-7.03).

The Jonas danger sign is ``dG/dt > 0`` *sustained*, not a single positive month.
Two rules, both required:

1. **Run-length rule.** The trailing ``dG/dt`` has been positive for at least
   ``min_run`` consecutive months.
2. **HAC t-test.** Over the last ``months`` months, the annualised monthly
   increments ``12·ΔG`` have a mean significantly above 0 in a one-sided test with
   Newey–West (HAC) standard errors. The increments are serially correlated
   (composite indices are smoothed and interpolated), so an ordinary t-test would
   overstate significance. The default lag length is the Newey–West rule
   ``floor(4·(n/100)^(2/9))``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

__all__ = [
    "HacTest",
    "SustainedResult",
    "hac_mean_test",
    "newey_west_lags",
    "positive_run_length",
    "sustained_positive",
]


def newey_west_lags(n: int) -> int:
    """Newey–West (1994) rule-of-thumb lag length ``floor(4·(n/100)^(2/9))``."""
    return math.floor(4 * (n / 100) ** (2 / 9)) if n > 0 else 0


def positive_run_length(dgdt: pd.Series) -> int:
    """Number of consecutive positive values at the end of ``dgdt`` (NaNs dropped)."""
    values = dgdt.dropna().to_numpy(dtype="float64")
    nonpos = np.flatnonzero(values <= 0)
    return int(len(values) if nonpos.size == 0 else len(values) - 1 - nonpos[-1])


@dataclass(frozen=True)
class HacTest:
    """One-sided HAC test of ``mean > 0``."""

    mean: float
    se: float
    t_stat: float
    p_value: float
    n: int
    maxlags: int


def hac_mean_test(x: pd.Series | np.ndarray[Any, Any], maxlags: int | None = None) -> HacTest:
    """Test ``H1: E[x] > 0`` with a Newey–West standard error.

    Args:
        x: Observations (NaNs dropped).
        maxlags: HAC lag length; defaults to :func:`newey_west_lags`.
    """
    arr = np.asarray(pd.Series(x).dropna(), dtype="float64")
    n = arr.size
    if n < 3:
        raise ValueError("HAC test needs at least 3 observations")
    lags = newey_west_lags(n) if maxlags is None else maxlags
    fit = sm.OLS(arr, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    mean = float(fit.params[0])
    se = float(fit.bse[0])
    if se > 0:
        t = mean / se
        p = float(stats.norm.sf(t))
    else:  # constant increments: the sign is certain
        t = math.copysign(math.inf, mean) if mean != 0 else 0.0
        p = 0.0 if mean > 0 else 1.0
    return HacTest(mean=mean, se=se, t_stat=float(t), p_value=p, n=n, maxlags=lags)


@dataclass(frozen=True)
class SustainedResult:
    """Outcome of :func:`sustained_positive`.

    Attributes:
        run_length: Trailing months with ``dG/dt > 0``.
        min_run: Run length the rule requires.
        run_rule: ``run_length >= min_run``.
        mean_dgdt: Mean annualised increment over the test window (per year).
        se: Its Newey–West standard error.
        t_stat: ``mean_dgdt / se``.
        p_value: One-sided p-value for ``dG/dt > 0``.
        alpha: Significance level.
        hac_rule: ``p_value < alpha`` and ``mean_dgdt > 0``.
        n: Increments in the test window.
        maxlags: HAC lag length used.
        sustained: Both rules hold.
    """

    run_length: int
    min_run: int
    run_rule: bool
    mean_dgdt: float
    se: float
    t_stat: float
    p_value: float
    alpha: float
    hac_rule: bool
    n: int
    maxlags: int
    sustained: bool

    def as_dict(self) -> dict[str, Any]:
        """Plain ``dict`` for JSON and ``DialReading.detail``."""
        return asdict(self)


def sustained_positive(
    velocity: pd.DataFrame,
    *,
    min_run: int = 6,
    alpha: float = 0.05,
    months: int = 36,
    maxlags: int | None = None,
) -> SustainedResult:
    """Apply the run-length rule and the HAC t-test to a velocity frame.

    Args:
        velocity: Output of :func:`kairos.jonas.gap_velocity` (needs ``gap`` and
            ``dgdt``) or any frame with those columns in time order.
        min_run: Months of consecutive positive ``dG/dt`` required.
        alpha: One-sided significance level of the HAC test.
        months: Trailing months of ``G`` increments used by the HAC test.
        maxlags: HAC lag length (default: Newey–West rule).
    """
    if min_run < 1 or months < 3 or not 0 < alpha < 1:
        raise ValueError("need min_run >= 1, months >= 3 and 0 < alpha < 1")
    run = positive_run_length(velocity["dgdt"])
    increments = 12.0 * velocity["gap"].astype("float64").diff().dropna()
    test = hac_mean_test(increments.iloc[-months:], maxlags)
    run_rule = run >= min_run
    hac_rule = test.mean > 0 and test.p_value < alpha
    return SustainedResult(
        run_length=run,
        min_run=min_run,
        run_rule=run_rule,
        mean_dgdt=test.mean,
        se=test.se,
        t_stat=test.t_stat,
        p_value=test.p_value,
        alpha=alpha,
        hac_rule=hac_rule,
        n=test.n,
        maxlags=test.maxlags,
        sustained=run_rule and hac_rule,
    )
