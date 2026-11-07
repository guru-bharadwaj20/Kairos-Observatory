"""Vinge Horizon h*: how far ahead capability can still be forecast within tolerance.

Pipeline: :func:`rolling_origin_backtest` -> :func:`score_backtest` ->
:func:`error_horizon` / :func:`band_horizon` -> :func:`hstar_series` ->
:func:`shrinkage_test` -> :func:`vinge_dial`. See ``docs/vinge.md``.
"""

from kairos.vinge.backtest import (
    from_prophet_cv,
    history_at,
    monthly_cutoffs,
    prophet_cross_validation,
    rolling_origin_backtest,
)
from kairos.vinge.baselines import LogDriftForecaster
from kairos.vinge.dial import (
    VingeThresholds,
    load_thresholds,
    spec_from_thresholds,
    vinge_dial,
)
from kairos.vinge.horizon import HorizonResult, band_horizon, error_horizon
from kairos.vinge.metrics import crps_gaussian, crps_quantiles, error_curve, score_backtest
from kairos.vinge.neural import (
    CrossCheck,
    QuantileAdapter,
    QuantileForecaster,
    cross_check,
    from_neuralprophet_columns,
)
from kairos.vinge.schema import (
    BacktestFrame,
    Forecaster,
    ForecastFrame,
    validate_backtest_frame,
    validate_forecast_frame,
)
from kairos.vinge.series import HorizonSpec, compute_hstar_series, hstar_series
from kairos.vinge.trend import ShrinkageResult, shrinkage_by_series, shrinkage_test

__all__ = [
    "BacktestFrame",
    "CrossCheck",
    "ForecastFrame",
    "Forecaster",
    "HorizonResult",
    "HorizonSpec",
    "LogDriftForecaster",
    "QuantileAdapter",
    "QuantileForecaster",
    "ShrinkageResult",
    "VingeThresholds",
    "band_horizon",
    "compute_hstar_series",
    "cross_check",
    "crps_gaussian",
    "crps_quantiles",
    "error_curve",
    "error_horizon",
    "from_neuralprophet_columns",
    "from_prophet_cv",
    "history_at",
    "hstar_series",
    "load_thresholds",
    "monthly_cutoffs",
    "prophet_cross_validation",
    "rolling_origin_backtest",
    "score_backtest",
    "shrinkage_by_series",
    "shrinkage_test",
    "spec_from_thresholds",
    "validate_backtest_frame",
    "validate_forecast_frame",
    "vinge_dial",
]
