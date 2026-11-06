"""NeuralProphet wrapper: one place that builds, quiets and fits every model.

* :class:`NeuralConfig` holds the hyper-parameters KAIROS varies (lags, horizon,
  epochs, quantiles, regularisation). It is a frozen pydantic model, so it is
  hashable, picklable for :func:`kairos.compute.pmap` and part of fit-cache keys.
* :func:`trainer_config` turns a :class:`torch.device` from
  :func:`kairos.compute.get_device` into the PyTorch Lightning trainer settings
  NeuralProphet forwards to ``pl.Trainer`` (accelerator, devices, no progress bar,
  no checkpoints, no ``lightning_logs/`` in the working directory).
* :func:`quiet_neuralprophet` silences NeuralProphet, Lightning and their
  warnings for the duration of a fit.
* :func:`fit_model` fits with early stopping on a validation frame when one is
  given (patience 20 epochs, NeuralProphet's default) and seeds every RNG first.

NeuralProphet is imported lazily: importing :mod:`kairos.neural` stays cheap.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kairos.types import SeriesFrame, Transform, validate_series_frame

if TYPE_CHECKING:
    import torch
    from neuralprophet import NeuralProphet

__all__ = [
    "NeuralConfig",
    "build_model",
    "fit_model",
    "predict",
    "quiet_neuralprophet",
    "to_np_frame",
    "trainer_config",
]

_NOISY_LOGGERS = (
    "NP",
    "neuralprophet",
    "lightning",
    "lightning.pytorch",
    "lightning.fabric",
    "lightning_fabric",
    "pytorch_lightning",
)


class NeuralConfig(BaseModel):
    """Hyper-parameters of one NeuralProphet model.

    Defaults suit ~200 monthly points (CONTRIBUTING D9): a small linear AR-Net,
    no seasonality (the series are annual-scale trends), a modest learning rate
    and early stopping.

    Attributes:
        n_lags: Autoregressive lags of the target (``0`` = no AR-Net).
        n_forecasts: Forecast horizon in steps (months).
        epochs: Maximum training epochs (early stopping may end sooner).
        batch_size: Mini-batch size.
        learning_rate: AdamW learning rate.
        quantiles: Extra quantiles to forecast (the median is the point forecast).
        n_changepoints: Trend changepoints.
        trend_reg: Trend regularisation strength.
        ar_reg: AR sparsity regularisation (``None`` = off).
        growth: Trend type.
        early_stopping: Stop when the validation (or training) loss stalls.
        loss_func: NeuralProphet loss name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_lags: int = Field(default=0, ge=0)
    n_forecasts: int = Field(default=1, ge=1)
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    quantiles: tuple[float, ...] = ()
    n_changepoints: int = Field(default=5, ge=0)
    trend_reg: float = Field(default=0.0, ge=0)
    ar_reg: float | None = None
    growth: Literal["off", "linear", "discontinuous"] = "linear"
    early_stopping: bool = True
    loss_func: str = "SmoothL1Loss"

    @field_validator("quantiles")
    @classmethod
    def _valid_quantiles(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0.0 < q < 1.0 for q in value):
            raise ValueError("quantiles must lie strictly between 0 and 1")
        return tuple(sorted(set(value)))


def trainer_config(device: torch.device | str | None = None) -> dict[str, Any]:
    """Lightning trainer settings for ``device``.

    Args:
        device: Target device; ``None`` asks :func:`kairos.compute.get_device`
            (``Settings.device``, ``"auto"`` by default).

    Returns:
        A dict for ``NeuralProphet(trainer_config=...)``: ``accelerator`` and
        ``devices`` from the device, plus no progress bar, model summary,
        checkpoints or logs in the working directory.
    """
    import torch

    from kairos.compute import get_device

    dev = get_device() if device is None else torch.device(device)
    if dev.type == "cuda":
        accel: dict[str, Any] = {"accelerator": "gpu", "devices": [dev.index or 0]}
    else:
        accel = {"accelerator": "cpu", "devices": 1}
    return {
        **accel,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "enable_checkpointing": False,
        "default_root_dir": str(_log_root()),
    }


def _log_root() -> Path:
    root = Path(tempfile.gettempdir()) / "kairos-lightning"
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextlib.contextmanager
def quiet_neuralprophet() -> Iterator[None]:
    """Silence NeuralProphet and Lightning logs and warnings inside the block."""
    previous = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
    try:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


def build_model(config: NeuralConfig, device: torch.device | str | None = None) -> NeuralProphet:
    """Create an unfitted NeuralProphet model from ``config`` on ``device``."""
    with quiet_neuralprophet():
        import neuralprophet
        from neuralprophet.logger import MetricsLogger

        neuralprophet.set_log_level("ERROR")
        model = neuralprophet.NeuralProphet(
            growth=config.growth,
            n_changepoints=config.n_changepoints,
            trend_reg=config.trend_reg,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            n_lags=config.n_lags,
            n_forecasts=config.n_forecasts,
            ar_reg=config.ar_reg,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            loss_func=config.loss_func,
            quantiles=list(config.quantiles),
            accelerator=None,
            trainer_config=trainer_config(device),
        )
        # NeuralProphet logs metrics to TensorBoard under the working directory by default.
        model.metrics_logger = MetricsLogger(save_dir=str(_log_root()))
        return model


def fit_model(
    model: NeuralProphet,
    train: pd.DataFrame,
    *,
    config: NeuralConfig,
    validation: pd.DataFrame | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Seed every RNG and fit ``model`` quietly.

    Args:
        model: Model from :func:`build_model` (regressors and events already added).
        train: NeuralProphet frame (``ds``, ``y`` and regressor columns).
        config: The config the model was built from (controls early stopping).
        validation: Optional held-out frame; early stopping then watches its loss.
        seed: Seed for this fit; ``None`` uses ``Settings.seed``.

    Returns:
        NeuralProphet's per-epoch metrics frame.
    """
    from kairos.compute import seed_everything

    state = seed_everything(seed)
    with quiet_neuralprophet():
        import neuralprophet

        neuralprophet.set_random_seed(state.seed)
        metrics = model.fit(
            train,
            freq="MS",
            validation_df=validation,
            early_stopping=config.early_stopping,
            progress=None,
        )
    return metrics if isinstance(metrics, pd.DataFrame) else pd.DataFrame()


def predict(model: NeuralProphet, df: pd.DataFrame) -> pd.DataFrame:
    """Quiet ``model.predict(df)``."""
    with quiet_neuralprophet():
        out = model.predict(df)
    return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)


def to_np_frame(
    frame: SeriesFrame, series_id: str, *, transform: Transform = "log", column: str = "y"
) -> pd.DataFrame:
    """Extract one series from a ``SeriesFrame`` as a NeuralProphet ``ds``/``column`` frame.

    Args:
        frame: Long ``SeriesFrame`` (raw positive units).
        series_id: Series to extract.
        transform: ``"log"`` (default, growth series), ``"logit"`` (shares) or ``"none"``.
        column: Name of the value column in the result.

    Returns:
        A frame with ``ds`` (sorted, month start) and ``column`` (transformed).

    Raises:
        KeyError: If ``series_id`` is not in ``frame``.
    """
    validate_series_frame(frame, positive=transform == "log")
    sub = frame.loc[frame["series_id"] == series_id, ["ds", "y"]]
    if sub.empty:
        raise KeyError(f"series {series_id!r} not in frame")
    values = sub["y"].to_numpy(dtype=np.float64)
    if transform == "log":
        values = np.log(values)
    elif transform == "logit":
        values = np.log(values / (1.0 - values))
    out = pd.DataFrame({"ds": sub["ds"].to_numpy(), column: values})
    return out.sort_values("ds", ignore_index=True)
