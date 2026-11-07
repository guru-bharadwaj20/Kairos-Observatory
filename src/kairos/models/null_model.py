"""Prophet null model of "normal" progress, one series at a time.

The null model is Prophet on ``ln y`` (:mod:`kairos.models.logscale`): piecewise
exponential growth with sparse changepoints. It stands for the *boring world*;
the detectors in this package look for reality beating it.

Configuration
-------------
:class:`ProphetConfig` resolves, first wins: keyword arguments, ``KAIROS_PROPHET_*``
environment variables, the ``prophet:`` section of the KAIROS YAML file
(:func:`kairos.settings.config_file`), then the defaults below::

    # kairos.yaml
    prophet:
      changepoint_prior_scale: 0.3
      n_changepoints: 25

Frontier series (running-maximum envelopes of the state of the art, D7) never get
yearly seasonality: a frontier has no calendar cycle, and fitting one only lets
release-date clustering leak into the trend.
"""

from __future__ import annotations

import fnmatch
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from kairos.compute.seeds import derive_seed, stan_seed
from kairos.models.logscale import LogProphet
from kairos.types import SeriesFrame, validate_series_frame

if TYPE_CHECKING:
    from prophet import Prophet

__all__ = [
    "DAYS_PER_YEAR",
    "NullModelFit",
    "ProphetConfig",
    "fit_null_model",
    "get_prophet_config",
    "is_frontier",
    "numpy_seed",
    "rates_per_year",
    "select_series",
    "series_seed",
]

DAYS_PER_YEAR = 365.25


class ProphetConfig(BaseSettings):
    """Prophet null-model settings.

    Attributes:
        changepoint_prior_scale: Laplace prior scale on slope changes (idea §9a: 0.3).
        n_changepoints: Potential changepoints placed uniformly in the history.
        changepoint_range: Share of the history in which changepoints may sit;
            0.9 (Prophet: 0.8) so 2020-23 shifts are reachable in a 2000-2026 series.
        interval_width: Width of the predictive interval (0.9 = 5 %..95 %).
        uncertainty_samples: Monte Carlo draws for the trend interval.
        horizon_months: Default forecast horizon.
        yearly_seasonality: Yearly seasonality for **non-frontier** series.
        frontier_patterns: ``fnmatch`` patterns of ``series_id`` treated as frontier.
    """

    model_config = SettingsConfigDict(env_prefix="KAIROS_PROPHET_", extra="ignore", frozen=True)

    changepoint_prior_scale: float = Field(default=0.3, gt=0)
    n_changepoints: int = Field(default=25, ge=0)
    changepoint_range: float = Field(default=0.9, gt=0, le=1)
    interval_width: float = Field(default=0.9, gt=0, lt=1)
    uncertainty_samples: int = Field(default=1000, ge=0)
    horizon_months: int = Field(default=36, ge=0)
    yearly_seasonality: bool | Literal["auto"] = "auto"
    frontier_patterns: tuple[str, ...] = ("*frontier*",)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Read the ``prophet:`` section of the KAIROS YAML file after the environment."""
        from kairos.settings import config_file

        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        path = config_file()
        if path is not None:
            sources.append(
                YamlConfigSettingsSource(
                    settings_cls, yaml_file=path, yaml_config_section="prophet"
                )
            )
        return tuple(sources)

    def prophet_kwargs(self, *, frontier: bool) -> dict[str, Any]:
        """Keyword arguments for :class:`prophet.Prophet` (monthly data: no weekly/daily)."""
        return {
            "growth": "linear",
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "interval_width": self.interval_width,
            "uncertainty_samples": self.uncertainty_samples,
            "yearly_seasonality": False if frontier else self.yearly_seasonality,
            "weekly_seasonality": False,
            "daily_seasonality": False,
        }


def get_prophet_config(**overrides: Any) -> ProphetConfig:
    """Return a :class:`ProphetConfig`, with ``overrides`` taking precedence."""
    return ProphetConfig(**overrides)


def is_frontier(series_id: str, config: ProphetConfig | None = None) -> bool:
    """Whether ``series_id`` matches one of the configured frontier patterns."""
    patterns = (config or ProphetConfig()).frontier_patterns
    return any(fnmatch.fnmatchcase(series_id, p) for p in patterns)


def series_seed(seed: int | None, series_id: str, *keys: object) -> int:
    """CmdStan seed for one series (stable across processes and platforms)."""
    if seed is None:
        from kairos.settings import get_settings

        seed = get_settings().seed
    return stan_seed(derive_seed(seed, "prophet", series_id, *keys))


def select_series(frame: SeriesFrame, series_id: str | None = None) -> tuple[str, pd.DataFrame]:
    """Return ``(series_id, rows)`` for one series of a long ``SeriesFrame``, sorted by ``ds``.

    Raises:
        ValueError: If ``series_id`` is ``None`` but the frame holds several series,
            or if the series is absent or has fewer than two observations.
    """
    validate_series_frame(frame)
    ids = pd.unique(frame["series_id"])
    if series_id is None:
        if len(ids) != 1:
            raise ValueError(f"frame holds {len(ids)} series; pass series_id")
        series_id = str(ids[0])
    rows = frame.loc[frame["series_id"] == series_id].sort_values("ds").reset_index(drop=True)
    if rows["y"].notna().sum() < 2:
        raise ValueError(f"series {series_id!r} needs at least two observations")
    return series_id, rows


@contextmanager
def numpy_seed(seed: int) -> Iterator[None]:
    """Seed NumPy's legacy global state (Prophet's predictive draws) and restore it."""
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed % 2**32)  # noqa: NPY002 - Prophet samples from the global state
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


@dataclass
class NullModelFit:
    """A fitted null model and the quantities the detectors need.

    Rates are in natural-log units **per year** (0.69 = doubling every year).

    Attributes:
        series_id: Fitted series.
        history: ``ds, y`` of the fitted rows, raw units.
        forecast: Back-transformed forecast over history plus horizon (raw units,
            log-scale copies in ``*_log`` columns).
        changepoints: Potential changepoint dates.
        deltas: Slope change at each changepoint (ln units per year).
        base_rate: Growth rate before the first changepoint (ln units per year).
        frontier: Whether the series was treated as a frontier series.
        seed: CmdStan seed used.
        fit_seconds: Wall time of fit plus predict.
        params: Prophet keyword arguments used.
        model: The fitted :class:`~kairos.models.logscale.LogProphet`, unless dropped
            (e.g. before returning from a worker).
    """

    series_id: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    changepoints: pd.DatetimeIndex
    deltas: np.ndarray[Any, np.dtype[np.float64]]
    base_rate: float
    frontier: bool
    seed: int
    fit_seconds: float
    params: dict[str, Any] = field(default_factory=dict)
    model: LogProphet | None = None

    @property
    def rates(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Growth rate after each changepoint (ln units per year)."""
        return np.asarray(self.base_rate + np.cumsum(self.deltas), dtype="float64")

    @property
    def final_rate(self) -> float:
        """Growth rate at the end of the history (ln units per year)."""
        return float(self.base_rate + self.deltas.sum())

    def without_model(self) -> NullModelFit:
        """Copy without the Prophet object (light and safely picklable)."""
        return NullModelFit(
            self.series_id,
            self.history,
            self.forecast,
            self.changepoints,
            self.deltas,
            self.base_rate,
            self.frontier,
            self.seed,
            self.fit_seconds,
            dict(self.params),
            None,
        )


def rates_per_year(model: Prophet) -> tuple[float, np.ndarray[Any, np.dtype[np.float64]]]:
    """Base rate ``k`` and changepoint ``delta`` in ln-y units per year."""
    params = model.params
    t_scale_years = pd.Timedelta(model.t_scale).total_seconds() / 86400.0 / DAYS_PER_YEAR
    if model.y_scale is None:
        raise ValueError("model is not fitted")
    factor = float(model.y_scale) / t_scale_years
    k = float(np.asarray(params["k"]).reshape(-1)[0]) * factor
    delta = np.asarray(params["delta"], dtype="float64")
    delta = delta[0] if delta.ndim == 2 else delta.reshape(-1)
    return k, np.asarray(delta * factor, dtype="float64")


def fit_null_model(
    frame: SeriesFrame,
    series_id: str | None = None,
    *,
    config: ProphetConfig | None = None,
    frontier: bool | None = None,
    seed: int | None = None,
    horizon: int | None = None,
    keep_model: bool = True,
) -> NullModelFit:
    """Fit the Prophet null model to one series.

    Args:
        frame: ``SeriesFrame`` in raw positive units (may hold several series).
        series_id: Series to fit; optional when ``frame`` holds one series.
        config: Model settings; default :class:`ProphetConfig` from config.
        frontier: Force frontier handling; ``None`` uses ``config.frontier_patterns``.
        seed: Master seed; the CmdStan seed is derived per series (:func:`series_seed`).
        horizon: Months to forecast past the history (default ``config.horizon_months``).
        keep_model: Keep the fitted Prophet object on the result.

    Returns:
        A :class:`NullModelFit`.
    """
    config = config or ProphetConfig()
    series_id, rows = select_series(frame, series_id)
    frontier = is_frontier(series_id, config) if frontier is None else frontier
    horizon = config.horizon_months if horizon is None else horizon
    kwargs = config.prophet_kwargs(frontier=frontier)
    fit_seed = series_seed(seed, series_id)

    start = time.perf_counter()
    model = LogProphet(**kwargs).fit(rows[["ds", "y"]], seed=fit_seed)
    with numpy_seed(fit_seed):
        forecast = model.predict(model.make_future_dataframe(horizon))
    elapsed = time.perf_counter() - start

    base_rate, deltas = rates_per_year(model.model)
    changepoints = pd.DatetimeIndex(model.model.changepoints)
    return NullModelFit(
        series_id=series_id,
        history=rows[["ds", "y"]].copy(),
        forecast=forecast,
        changepoints=changepoints,
        deltas=deltas,
        base_rate=base_rate,
        frontier=frontier,
        seed=fit_seed,
        fit_seconds=elapsed,
        params=kwargs,
        model=model if keep_model else None,
    )
