"""Logistic null model with a time-varying cap from the human-capacity index ``H(t)``.

Idea §5: model capability ``C`` as bounded by oversight capacity, ``cap = H(t)``.

Units (D8)
----------
``C`` and ``H`` are in different units, so only a normalised comparison means
anything. Both are divided by their value at a common ``base`` month, and the
model works on the log scale::

    y     = ln(C / C(base))
    cap   = ln(H / H(base)) + ln(headroom)
    floor = min(y) - floor_margin

``headroom`` (default 2) is how far capability may sit above oversight capacity,
relative to the base month, before the cap binds. Future values of the cap come
from a log-linear trend fitted to the last ``extrapolate_months`` of ``H``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from kairos.models.logscale import LogProphet
from kairos.models.null_model import (
    NullModelFit,
    ProphetConfig,
    numpy_seed,
    rates_per_year,
    select_series,
    series_seed,
)
from kairos.types import SeriesFrame

__all__ = ["CappedFit", "build_cap", "fit_capped_model"]


def build_cap(
    h: pd.DataFrame,
    ds: pd.Series | pd.DatetimeIndex,
    *,
    base: pd.Timestamp,
    headroom: float = 2.0,
    extrapolate_months: int = 24,
) -> pd.Series:
    """Cap ``ln(H/H(base)) + ln(headroom)`` on the dates ``ds``.

    Dates inside the ``H`` history are interpolated in log space; later dates follow
    a least-squares log-linear trend over the last ``extrapolate_months`` of ``H``;
    earlier dates hold the first value.

    Args:
        h: Rows of one ``H`` series (``ds``, raw positive ``y``).
        ds: Dates to evaluate the cap on.
        base: Normalisation month (must lie inside the ``H`` history).
        headroom: Multiplicative allowance of ``C`` over ``H`` at the base month.
        extrapolate_months: Window of the trend used past the end of ``H``.
    """
    if headroom <= 0:
        raise ValueError("headroom must be positive")
    h = h.dropna(subset=["y"]).sort_values("ds")
    t_h = h["ds"].astype("int64").to_numpy().astype("float64")
    log_h = np.log(h["y"].to_numpy(dtype="float64"))
    t_base = float(pd.Timestamp(base).value)
    if not t_h[0] <= t_base <= t_h[-1]:
        raise ValueError("base month must lie inside the H history")
    h_base = float(np.interp(t_base, t_h, log_h))

    t = pd.DatetimeIndex(ds).asi8.astype("float64")
    out = np.interp(t, t_h, log_h)  # clamps outside the range
    after = t > t_h[-1]
    if after.any():
        window = max(2, min(extrapolate_months, len(t_h)))
        slope, intercept = np.polyfit(t_h[-window:], log_h[-window:], 1)
        anchor = log_h[-1] - (slope * t_h[-1] + intercept)  # continue from the last value
        out[after] = slope * t[after] + intercept + anchor
    return pd.Series(out - h_base + math.log(headroom), index=pd.DatetimeIndex(ds), name="cap")


@dataclass
class CappedFit:
    """A logistic null model of ``C`` capped by ``H``.

    Attributes:
        c_id: Capability series.
        h_id: Human-capacity series.
        base: Normalisation month.
        headroom: Cap allowance at the base month.
        forecast: Log-normalised frame ``ds, y, yhat, yhat_lower, yhat_upper, cap, floor``
            (``y`` observed, missing in the future).
        fit: Changepoint view of the logistic fit (rates are only indicative under
            logistic growth).
        params: Prophet keyword arguments used.
    """

    c_id: str
    h_id: str
    base: pd.Timestamp
    headroom: float
    forecast: pd.DataFrame
    fit: NullModelFit
    params: dict[str, Any] = field(default_factory=dict)


def fit_capped_model(
    c: SeriesFrame,
    h: SeriesFrame,
    *,
    c_id: str | None = None,
    h_id: str | None = None,
    base: str | pd.Timestamp | None = None,
    headroom: float = 2.0,
    floor_margin: float = 1.0,
    extrapolate_months: int = 24,
    config: ProphetConfig | None = None,
    seed: int | None = None,
    horizon: int | None = None,
) -> CappedFit:
    """Fit ``growth="logistic"`` to ``ln C`` with ``cap = ln H(t)`` (normalised).

    Args:
        c: Frame holding the capability series (raw units).
        h: Frame holding the human-capacity index (raw units).
        c_id: Capability ``series_id`` (optional for a one-series frame).
        h_id: Human-capacity ``series_id`` (optional for a one-series frame).
        base: Normalisation month (default: first month both series cover).
        headroom: See module docstring.
        floor_margin: Log units below the lowest observation for the logistic floor.
        extrapolate_months: Trend window for future cap values.
        config: Null-model settings (priors, interval width).
        seed: Master seed.
        horizon: Forecast months (default ``config.horizon_months``).
    """
    config = config or ProphetConfig()
    c_id, c_rows = select_series(c, c_id)
    h_id, h_rows = select_series(h, h_id)
    if base is None:
        base = max(c_rows["ds"].min(), h_rows["ds"].min())
    base = pd.Timestamp(base)
    c_obs = c_rows.dropna(subset=["y"])
    if base not in set(c_obs["ds"]):
        raise ValueError(f"C has no observation at base month {base:%Y-%m}")
    c_base = float(c_obs.loc[c_obs["ds"] == base, "y"].iloc[0])

    horizon = config.horizon_months if horizon is None else horizon
    kwargs = config.prophet_kwargs(frontier=True) | {"growth": "logistic"}
    data = pd.DataFrame({"ds": c_rows["ds"], "y": c_rows["y"] / c_base})
    log_y = np.log(data["y"])
    floor = float(log_y.min()) - floor_margin
    data["cap"] = build_cap(
        h_rows, data["ds"], base=base, headroom=headroom, extrapolate_months=extrapolate_months
    ).to_numpy()
    data["floor"] = floor
    if (data["cap"] <= floor).any():
        raise ValueError("cap falls to or below the floor; raise floor_margin or headroom")

    fit_seed = series_seed(seed, c_id, "logistic", h_id)
    start = time.perf_counter()
    model = LogProphet(**kwargs).fit(data, seed=fit_seed)
    future = model.make_future_dataframe(horizon)
    future["cap"] = build_cap(
        h_rows, future["ds"], base=base, headroom=headroom, extrapolate_months=extrapolate_months
    ).to_numpy()
    future["floor"] = floor
    with numpy_seed(fit_seed):
        raw = model.predict_log(future)
    elapsed = time.perf_counter() - start

    forecast = pd.DataFrame(
        {
            "ds": raw["ds"].to_numpy(),
            "y": np.nan,
            "yhat": raw["yhat"].to_numpy(),
            "yhat_lower": raw["yhat_lower"].to_numpy(),
            "yhat_upper": raw["yhat_upper"].to_numpy(),
            "cap": future["cap"].to_numpy(),
            "floor": floor,
        }
    )
    forecast.loc[: len(data) - 1, "y"] = log_y.to_numpy()
    base_rate, deltas = rates_per_year(model.model)
    view = NullModelFit(
        series_id=c_id,
        history=c_rows[["ds", "y"]].copy(),
        forecast=forecast,
        changepoints=pd.DatetimeIndex(model.model.changepoints),
        deltas=deltas,
        base_rate=base_rate,
        frontier=True,
        seed=fit_seed,
        fit_seconds=elapsed,
        params=kwargs,
    )
    return CappedFit(c_id, h_id, base, headroom, forecast, view, kwargs)
