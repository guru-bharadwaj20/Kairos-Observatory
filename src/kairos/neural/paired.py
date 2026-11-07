"""Paired models for ``κ``: exogenous-only versus exogenous plus self-lags.

Both models share everything (config, drivers, events, seed, train/test split)
except one input channel:

* ``"exo"``: trend plus lagged drivers only;
* ``"self"``: the same plus the target's own last ``self_lags`` values;
* ``"null"``: the same as ``"self"``, but the self channel is a **block-shuffled
  surrogate** of the target (CONTRIBUTING D9, the permutation null).

The self channel is fed to NeuralProphet as a lagged regressor holding a copy
of ``y``. With linear layers (the default) this is the same function class as
the AR-Net (``n_lags``): a linear map of the last ``self_lags`` values of ``y``
onto every horizon. Using the lagged-regressor slot for both the real and the
shuffled channel makes the null *exactly* the same architecture as the
``"self"`` model, with only the alignment of the input destroyed.

The surrogate permutes blocks of first differences of ``y`` and re-integrates
them from ``y[0]``: short-range autocorrelation and the overall growth are kept,
the time alignment with the target is not.

Each fit is a picklable :class:`FitTask` run by the module-level :func:`fit_task`,
so the ensemble runner (K-4.07) can push many of them through
:func:`kairos.compute.pmap` on Windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd

from kairos.neural.evaluation import holdout_split, horizon_errors
from kairos.neural.events import EventSpec, add_events, with_events
from kairos.neural.model import NeuralConfig, build_model, fit_model, predict
from kairos.neural.regressors import DEFAULT_DRIVERS, DriverSpec, add_lagged_drivers

__all__ = [
    "SELF_COLUMN",
    "FitResult",
    "FitTask",
    "PairedSpec",
    "Variant",
    "block_shuffled",
    "fit_task",
    "paired_tasks",
    "run_paired",
]

SELF_COLUMN = "self_lag"
Variant = Literal["exo", "self", "null"]


@dataclass(frozen=True)
class PairedSpec:
    """Everything the paired models share.

    Attributes:
        config: NeuralProphet config (``n_lags`` must be 0; self-lags come from
            ``self_lags``). ``n_forecasts`` sets the horizons scored.
        drivers: Lagged exogenous drivers.
        self_lags: Window of the target's own lags in the ``"self"`` model.
        test_months: Hold-out months at the end of the series.
        null_block: Block length (months) of the shuffled surrogate.
        events: Optional events added to every variant.
    """

    config: NeuralConfig = field(
        default_factory=lambda: NeuralConfig(
            n_forecasts=6,
            epochs=300,
            batch_size=16,
            learning_rate=0.1,
            n_changepoints=0,
            early_stopping=False,
        )
    )
    drivers: tuple[DriverSpec, ...] = DEFAULT_DRIVERS
    self_lags: int = 12
    test_months: int = 24
    null_block: int = 12
    events: tuple[EventSpec, ...] = ()

    def __post_init__(self) -> None:
        """Run post init."""
        if self.config.n_lags != 0:
            raise ValueError("config.n_lags must be 0; set PairedSpec.self_lags instead")
        if self.self_lags < 1 or self.null_block < 1:
            raise ValueError("self_lags and null_block must be >= 1")


@dataclass(frozen=True)
class FitTask:
    """One fit of one variant (picklable).

    Attributes:
        design: Wide frame ``ds, y, <driver columns>`` (from ``design_frame``).
        spec: Shared settings.
        variant: Which model to fit.
        seed: Training seed (shared by the paired ``exo``/``self`` fits).
        perm: Index of the null permutation (``"null"`` only).
        perm_seed: Seed of the surrogate shuffle (``"null"`` only).
        device: Device string for :func:`kairos.neural.build_model`.
        key: Free-form label carried into the result (e.g. a rolling window end).
    """

    design: pd.DataFrame
    spec: PairedSpec
    variant: Variant
    seed: int
    perm: int = -1
    perm_seed: int = 0
    device: str = "cpu"
    key: str = ""


@dataclass(frozen=True)
class FitResult:
    """Out-of-sample skill of one fit.

    Attributes:
        variant: Model variant.
        seed: Training seed.
        perm: Null permutation index (``-1`` otherwise).
        key: Label copied from the task.
        errors: MAE per horizon on the hold-out window (log units).
    """

    variant: Variant
    seed: int
    perm: int
    key: str
    errors: tuple[float, ...]

    @property
    def mae(self) -> float:
        """MAE averaged over horizons."""
        return float(np.mean(self.errors))


def block_shuffled(
    values: np.ndarray[tuple[int], np.dtype[np.float64]], block: int, seed: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Surrogate of ``values``: permute blocks of its first differences, re-integrate."""
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    diffs = np.diff(x)
    blocks = [diffs[i : i + block] for i in range(0, len(diffs), block)]
    order = np.random.default_rng(seed).permutation(len(blocks))
    shuffled = np.concatenate([blocks[i] for i in order])
    return np.concatenate([[x[0]], x[0] + np.cumsum(shuffled)])


def _variant_frame(task: FitTask) -> pd.DataFrame:
    cols = ["ds", "y", *(d.column for d in task.spec.drivers)]
    df = task.design[cols].sort_values("ds").reset_index(drop=True)
    if task.variant == "self":
        df[SELF_COLUMN] = df["y"].to_numpy()
    elif task.variant == "null":
        df[SELF_COLUMN] = block_shuffled(df["y"].to_numpy(), task.spec.null_block, task.perm_seed)
    return df


def fit_task(task: FitTask) -> FitResult:
    """Fit one variant on the training part and score it on the hold-out window."""
    spec = task.spec
    df = _variant_frame(task)
    model = build_model(spec.config, task.device)
    add_lagged_drivers(model, spec.drivers)
    if task.variant != "exo":
        model.add_lagged_regressor(SELF_COLUMN, n_lags=spec.self_lags)
    if spec.events:
        add_events(model, spec.events)
        df = with_events(model, df, spec.events)
    split = holdout_split(df, spec.test_months)
    fit_model(model, split.train, config=spec.config, seed=task.seed)
    forecast = predict(model, split.full)
    errors = horizon_errors(forecast, spec.config.n_forecasts, split.test_start)
    return FitResult(task.variant, task.seed, task.perm, task.key, tuple(map(float, errors)))


def paired_tasks(
    design: pd.DataFrame,
    spec: PairedSpec,
    seeds: Sequence[int],
    *,
    device: str = "cpu",
    key: str = "",
) -> list[FitTask]:
    """One ``exo`` and one ``self`` task per seed, on identical data and splits."""
    base = FitTask(design, spec, "exo", 0, device=device, key=key)
    return [replace(base, variant=v, seed=s) for s in seeds for v in ("exo", "self")]


def run_paired(
    design: pd.DataFrame, spec: PairedSpec, seed: int = 0, *, device: str = "cpu"
) -> tuple[FitResult, FitResult]:
    """Fit the ``exo`` and ``self`` models in-process and return their results."""
    exo, self_ = (fit_task(t) for t in paired_tasks(design, spec, [seed], device=device))
    return exo, self_
