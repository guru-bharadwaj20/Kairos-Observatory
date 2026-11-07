"""K-9.05: COVID-19 in Italy, early 2020, with and without the lockdown event.

The question is whether an *event-conditioned* forecast sees the epidemic curve
bend, and whether it beats the no-event counterfactual on what happened next.

Model (log-linear with a lagged event hinge, the stand-in for NeuralProphet's
event regressors from K-4.03)::

    ln cases_w = a + b·t + c·max(0, t - (t_event + lag))

* **Conditional forecast**: fitted on the weeks up to the as-of date, with the
  national lockdown of 9 March 2020 as the event and a 14-day lag (incubation
  plus reporting delay, fixed a priori, not tuned).
* **Counterfactual**: the same data before the hinge, fitted with ``c = 0``,
  i.e. "no lockdown, the exponential continues".

Both are scored on the weeks after the as-of date by mean absolute log error
(a factor error, D2). Data: weekly new cases for Italy from Our World in Data
(WHO), CC BY 4.0. Weekly, so this case does not use the monthly ``SeriesFrame``.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd

from kairos.validation.cases import CaseResult
from kairos.validation.datasets import read_snapshot_csv

__all__ = ["LOCKDOWN", "event_design", "fit_event_model", "load_italy", "run"]

FloatArray = npt.NDArray[np.float64]

LOCKDOWN = date(2020, 3, 9)
LAG_DAYS = 14


def load_italy() -> pd.DataFrame:
    """Weekly new cases (``ds``, ``y``) for Italy, February to June 2020."""
    raw = read_snapshot_csv("covid_italy_weekly.csv")
    return pd.DataFrame({"ds": pd.to_datetime(raw["ds"]), "y": raw["new_cases"].astype(float)})


def event_design(ds: pd.Series, *, event: date = LOCKDOWN, lag_days: int = LAG_DAYS) -> FloatArray:
    """Design matrix ``[1, t, hinge]`` with ``t`` in weeks since the event."""
    weeks = (ds - pd.Timestamp(event)).dt.days.to_numpy(dtype=np.float64) / 7.0
    hinge = np.maximum(0.0, weeks - lag_days / 7.0)
    return np.column_stack([np.ones_like(weeks), weeks, hinge])


def fit_event_model(design: FloatArray, y: FloatArray) -> FloatArray:
    """Least-squares coefficients of ``ln y`` on ``design``."""
    coef, *_ = np.linalg.lstsq(design, np.log(y), rcond=None)
    return np.asarray(coef, dtype=np.float64)


def run(*, as_of: date = date(2020, 4, 5), start: date = date(2020, 2, 23)) -> CaseResult:
    """Fit both forecasts at ``as_of`` and score them on the following weeks."""
    data = load_italy()
    data = data[data["ds"] >= pd.Timestamp(start)].reset_index(drop=True)
    train = data[data["ds"] <= pd.Timestamp(as_of)]
    test = data[data["ds"] > pd.Timestamp(as_of)]
    x_train, x_test = event_design(train["ds"]), event_design(test["ds"])
    y_train, y_test = train["y"].to_numpy(), test["y"].to_numpy()

    coef = fit_event_model(x_train, y_train)
    pre = x_train[:, 2] == 0
    coef_cf = np.append(fit_event_model(x_train[pre, :2], y_train[pre]), 0.0)

    log_err = np.abs(x_test @ coef - np.log(y_test))
    log_err_cf = np.abs(x_test @ coef_cf - np.log(y_test))
    mae, mae_cf = float(log_err.mean()), float(log_err_cf.mean())
    bends = bool(coef[2] < 0 and coef[1] + coef[2] < 0)
    table = pd.DataFrame({
        "ds": test["ds"],
        "actual": y_test,
        "conditional": np.exp(x_test @ coef),
        "counterfactual": np.exp(x_test @ coef_cf),
    })  # fmt: skip
    return CaseResult(
        case="covid",
        passed=bool(bends and mae < mae_cf),
        criterion=(
            "lockdown-conditioned fit bends (post-event weekly growth < 0) and beats the "
            "no-event counterfactual on mean absolute log error"
        ),
        summary={
            "as_of": as_of.isoformat(),
            "weekly_growth_pre": round(float(coef[1]), 3),
            "weekly_growth_post": round(float(coef[1] + coef[2]), 3),
            "mae_log_conditional": round(mae, 3),
            "mae_log_counterfactual": round(mae_cf, 3),
            "factor_error_conditional": round(float(np.exp(mae)), 2),
            "factor_error_counterfactual": round(float(np.exp(mae_cf)), 2),
            "train_weeks": len(train),
            "test_weeks": len(test),
        },
        table=table,
        notes=[
            "Only two post-hinge weeks inform the bend at the as-of date; the slope "
            "is extrapolated linearly in log space and overshoots the decline.",
            "Case counts are testing-limited in March 2020; scores measure fit to "
            "reported cases, not infections.",
        ],
    )
