"""Classical cross-check of ``κ``: Granger-style tests with ``statsmodels``.

``κ`` asks whether the target's own past improves forecasts *after* the
drivers. The classical analogue is a nested F-test in an ARX regression on
first differences (log growth rates, so the series are close to stationary)::

    restricted:    Δy_t = c + Σ_k Σ_{l=1..L_k} b_kl Δx_k,t-l + ε_t
    unrestricted:  restricted + Σ_{l=1..p} a_l Δy_t-l

:func:`self_lag_f_test` tests ``a_1 = ... = a_p = 0``. :func:`var_granger`
adds the standard VAR Granger test of "drivers cause the target" for context.
:func:`agreement_table` lines both up with the ``κ`` permutation test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kairos.neural.kappa import KappaResult

__all__ = ["GrangerResult", "agreement_table", "self_lag_f_test", "var_granger"]


@dataclass(frozen=True)
class GrangerResult:
    """Result of an F-test.

    Attributes:
        f_stat: F statistic.
        p_value: p-value.
        df_num: Numerator degrees of freedom.
        df_denom: Denominator degrees of freedom.
        n_obs: Observations used.
        lags: Lag order tested.
    """

    f_stat: float
    p_value: float
    df_num: float
    df_denom: float
    n_obs: int
    lags: int


def _lagged(series: pd.Series, lags: int, name: str) -> pd.DataFrame:
    return pd.DataFrame({f"{name}_l{lag}": series.shift(lag) for lag in range(1, lags + 1)})


def self_lag_f_test(
    design: pd.DataFrame,
    drivers: Mapping[str, int],
    *,
    self_lags: int = 12,
    difference: bool = True,
) -> GrangerResult:
    """F-test of the target's own lags in an ARX regression on the drivers.

    Args:
        design: Wide frame with ``ds``, ``y`` and the driver columns.
        drivers: Driver column mapped to its lag window.
        self_lags: Own lags ``p`` tested.
        difference: Work on first differences (recommended for trending logs).

    Raises:
        ValueError: If too few observations remain for the unrestricted model.
    """
    import statsmodels.api as sm

    df = design.sort_values("ds").reset_index(drop=True)
    data = df[["y", *drivers]].astype(np.float64)
    if difference:
        data = data.diff()
    parts = [_lagged(data[col], lags, col) for col, lags in drivers.items()]
    own = _lagged(data["y"], self_lags, "y")
    frame = pd.concat([data["y"].rename("target"), *parts, own], axis=1).dropna()
    n_params = 1 + sum(drivers.values()) + self_lags
    if len(frame) <= n_params + 1:
        raise ValueError(f"{len(frame)} observations are too few for {n_params} parameters")
    endog = frame["target"]
    exog_r = sm.add_constant(frame.drop(columns=["target", *own.columns]), has_constant="add")
    exog_u = sm.add_constant(frame.drop(columns=["target"]), has_constant="add")
    restricted = sm.OLS(endog, exog_r).fit()
    unrestricted = sm.OLS(endog, exog_u).fit()
    f_stat, p_value, df_diff = unrestricted.compare_f_test(restricted)
    return GrangerResult(
        float(f_stat),
        float(p_value),
        float(df_diff),
        float(unrestricted.df_resid),
        len(frame),
        self_lags,
    )


def var_granger(
    design: pd.DataFrame,
    drivers: Sequence[str],
    *,
    maxlags: int = 12,
    difference: bool = True,
) -> GrangerResult:
    """VAR Granger test that ``drivers`` jointly cause ``y`` (lag order by AIC)."""
    from statsmodels.tsa.api import VAR

    data = design.sort_values("ds")[["y", *drivers]].astype(np.float64).reset_index(drop=True)
    if difference:
        data = data.diff().dropna()
    fitted = VAR(data).fit(maxlags=maxlags, ic="aic")
    lags = max(int(fitted.k_ar), 1)
    if fitted.k_ar == 0:
        fitted = VAR(data).fit(lags)
    test = fitted.test_causality("y", list(drivers), kind="f")
    df_num, df_denom = (float(v) for v in test.df)
    return GrangerResult(
        float(test.test_statistic), float(test.pvalue), df_num, df_denom, int(fitted.nobs), lags
    )


def agreement_table(
    rows: Mapping[str, tuple[KappaResult, GrangerResult]], *, alpha: float = 0.05
) -> pd.DataFrame:
    """Compare the ``κ`` permutation test with the classical self-lag F-test per series.

    Returns:
        One row per series: ``kappa``, ``kappa_p``, ``f_stat``, ``granger_p``,
        ``kappa_sig``, ``granger_sig`` and ``agree`` (both or neither significant).
    """
    records = []
    for name, (kappa, granger) in rows.items():
        kappa_sig = bool(np.isfinite(kappa.p_value) and kappa.p_value < alpha)
        granger_sig = bool(granger.p_value < alpha)
        records.append(
            {
                "series": name,
                "kappa": kappa.kappa,
                "kappa_p": kappa.p_value,
                "f_stat": granger.f_stat,
                "granger_p": granger.p_value,
                "kappa_sig": kappa_sig,
                "granger_sig": granger_sig,
                "agree": kappa_sig == granger_sig,
            }
        )
    return pd.DataFrame.from_records(records)
