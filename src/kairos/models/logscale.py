"""Log-scale Prophet wrapper.

Prophet's piecewise-linear trend on ``ln y`` is piecewise *exponential* growth in
raw units, which is the KAIROS null model of "normal" progress. This module fits
Prophet on ``ln y`` and back-transforms the forecast.

Back-transform
--------------
Prophet's predictive distribution on the log scale is symmetric, so ``yhat`` is
both its mean and its median. ``exp`` is monotone, so quantiles map exactly:
``exp(yhat)`` is the **median** of ``y`` (not its mean, which would need the
lognormal ``exp(sigma**2 / 2)`` correction) and ``exp(yhat_lower)`` /
``exp(yhat_upper)`` are the interval bounds in raw units.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from prophet import Prophet

__all__ = [
    "LOG_COLUMNS",
    "LogProphet",
    "back_transform",
    "quiet_prophet",
    "to_log_frame",
]

LOG_COLUMNS: Final = (
    "yhat",
    "yhat_lower",
    "yhat_upper",
    "trend",
    "trend_lower",
    "trend_upper",
)
"""Forecast columns that are levels on the log scale and are exponentiated."""

_NOISY_LOGGERS: Final = ("cmdstanpy", "prophet", "prophet.plot")


def quiet_prophet() -> None:
    """Silence Prophet and CmdStanPy chatter (safe to call in every worker)."""
    for name in _NOISY_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    warnings.filterwarnings("ignore", module="prophet")


def to_log_frame(frame: pd.DataFrame, *, extra: Iterable[str] = ()) -> pd.DataFrame:
    """Return a Prophet input frame ``ds, y = ln y`` from raw positive values.

    Args:
        frame: Frame with ``ds`` and strictly positive ``y`` (rows with missing
            ``y`` are kept; Prophet ignores them).
        extra: Further columns to carry over unchanged (e.g. ``cap``, ``floor``).

    Raises:
        ValueError: If any non-missing ``y`` is not strictly positive.
    """
    y = frame["y"].astype("float64")
    if not (y.dropna() > 0).all():
        raise ValueError("log-scale fit needs strictly positive y")
    out = pd.DataFrame({"ds": pd.to_datetime(frame["ds"]).to_numpy(), "y": np.log(y.to_numpy())})
    for column in extra:
        out[column] = frame[column].to_numpy()
    return out


def back_transform(
    forecast: pd.DataFrame, columns: Iterable[str] = LOG_COLUMNS, *, keep_log: bool = True
) -> pd.DataFrame:
    """Exponentiate log-scale forecast columns into raw units.

    Args:
        forecast: Prophet ``predict`` output fitted on ``ln y``.
        columns: Columns to exponentiate; missing ones are skipped.
        keep_log: Also keep the log-scale values as ``<column>_log``.

    Returns:
        A copy with the listed columns in raw units.
    """
    out = forecast.copy()
    for column in columns:
        if column not in out.columns:
            continue
        if keep_log:
            out[f"{column}_log"] = out[column]
        out[column] = np.exp(out[column].to_numpy(dtype="float64"))
    return out


class LogProphet:
    """A Prophet model fitted on ``ln y`` that forecasts in raw units.

    Args:
        **prophet_kwargs: Passed to :class:`prophet.Prophet` unchanged.

    Example::

        m = LogProphet(changepoint_prior_scale=0.3, interval_width=0.9)
        m.fit(series[["ds", "y"]], seed=7)
        raw = m.predict(m.make_future_dataframe(36))
    """

    def __init__(self, **prophet_kwargs: Any) -> None:
        quiet_prophet()
        from prophet import Prophet

        self.prophet_kwargs = dict(prophet_kwargs)
        self.model: Prophet = Prophet(**prophet_kwargs)
        self._extra: tuple[str, ...] = ()

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return getattr(self.model, "history", None) is not None

    def fit(self, frame: pd.DataFrame, *, seed: int | None = None, **kwargs: Any) -> LogProphet:
        """Fit on ``ln y``.

        Args:
            frame: ``ds`` and raw positive ``y``; ``cap``/``floor`` columns, when
                present, must already be on the log scale.
            seed: CmdStan seed (:attr:`kairos.compute.SeedState.stan_seed`).
            **kwargs: Passed to :meth:`prophet.Prophet.fit`.
        """
        self._extra = tuple(c for c in ("cap", "floor") if c in frame.columns)
        data = to_log_frame(frame, extra=self._extra)
        if seed is not None:
            kwargs["seed"] = int(seed)
        self.model.fit(data, **kwargs)
        return self

    def make_future_dataframe(
        self, periods: int, freq: str = "MS", *, include_history: bool = True
    ) -> pd.DataFrame:
        """Future ``ds`` frame (monthly by default)."""
        return self.model.make_future_dataframe(
            periods=periods, freq=freq, include_history=include_history
        )

    def predict_log(self, future: pd.DataFrame | None = None) -> pd.DataFrame:
        """Prophet forecast on the log scale."""
        return self.model.predict(future)

    def predict(self, future: pd.DataFrame | None = None, *, keep_log: bool = True) -> pd.DataFrame:
        """Forecast in raw units: ``exp`` of the median and interval bounds."""
        return back_transform(self.predict_log(future), keep_log=keep_log)
