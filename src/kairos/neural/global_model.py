"""Global NeuralProphet model across many C-side series, versus per-series models.

One global model sees every series (benchmarks, labs, countries) at once, with
an ``ID`` column, a per-series trend (``trend_global_local="local"``) and
per-series normalisation, while the AR weights are shared. Shared weights help
where single series are short. :func:`compare_global_local` scores the global
model and one local model per series on the same per-series hold-out.

Compute policy: the global model is the one wide job in Phase 4, so it runs on
:func:`kairos.compute.get_device` (the GPU when present); the many small local
models run on the CPU pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kairos.compute import pmap
from kairos.neural.evaluation import horizon_errors
from kairos.neural.model import NeuralConfig, build_model, fit_model, predict, to_np_frame
from kairos.types import SeriesFrame, Transform

__all__ = [
    "DEFAULT_GLOBAL_CONFIG",
    "compare_global_local",
    "fit_global",
    "global_frame",
]

DEFAULT_GLOBAL_CONFIG = NeuralConfig(
    n_lags=12,
    n_forecasts=6,
    epochs=300,
    batch_size=16,
    learning_rate=0.1,
    n_changepoints=0,
    early_stopping=False,
    trend_global_local="local",
)


def global_frame(
    frame: SeriesFrame, series_ids: Sequence[str], *, transform: Transform = "log"
) -> pd.DataFrame:
    """Stack series into NeuralProphet's multi-series layout ``ds, y, ID``."""
    parts = [to_np_frame(frame, sid, transform=transform).assign(ID=sid) for sid in series_ids]
    return pd.concat(parts, ignore_index=True)


def _split(df: pd.DataFrame, test_months: int) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    train, starts = [], {}
    for sid, part in df.groupby("ID", sort=False):
        part = part.sort_values("ds")
        if len(part) <= test_months:
            raise ValueError(f"series {sid!r} is shorter than test_months")
        train.append(part.iloc[: len(part) - test_months])
        starts[str(sid)] = pd.Timestamp(part["ds"].iloc[len(part) - test_months])
    return pd.concat(train, ignore_index=True), starts


def _score(
    forecast: pd.DataFrame, n_forecasts: int, starts: dict[str, pd.Timestamp]
) -> dict[str, float]:
    out = {}
    for sid, start in starts.items():
        part = forecast[forecast["ID"] == sid] if "ID" in forecast else forecast
        out[sid] = float(np.mean(horizon_errors(part, n_forecasts, start)))
    return out


def fit_global(
    df: pd.DataFrame,
    config: NeuralConfig = DEFAULT_GLOBAL_CONFIG,
    *,
    test_months: int = 24,
    device: str | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Fit one model on all series and return the hold-out MAE of each series."""
    train, starts = _split(df, test_months)
    model = build_model(config, device)
    fit_model(model, train, config=config, seed=seed)
    return _score(predict(model, df), config.n_forecasts, starts)


@dataclass(frozen=True)
class _LocalTask:
    df: pd.DataFrame
    config: NeuralConfig
    test_months: int
    seed: int | None


def _fit_local(task: _LocalTask) -> dict[str, float]:
    df = task.df.drop(columns="ID").reset_index(drop=True)
    sid = str(task.df["ID"].iloc[0])
    return {
        sid: fit_global(
            df.assign(ID=sid),
            task.config,
            test_months=task.test_months,
            device="cpu",
            seed=task.seed,
        )[sid]
    }


def compare_global_local(
    df: pd.DataFrame,
    config: NeuralConfig = DEFAULT_GLOBAL_CONFIG,
    *,
    test_months: int = 24,
    device: str | None = None,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Global model versus one model per series, on identical hold-outs.

    Returns:
        One row per series: ``series``, ``mae_global``, ``mae_local``, ``gain``
        (``1 - mae_global / mae_local``; positive when sharing helps) and ``better``.
    """
    glob = fit_global(df, config, test_months=test_months, device=device, seed=seed)
    tasks = [
        _LocalTask(part, config, test_months, seed) for _, part in df.groupby("ID", sort=False)
    ]
    local: dict[str, float] = {}
    for result in pmap(_fit_local, tasks, n_jobs=n_jobs):
        local.update(result)
    table = pd.DataFrame({"series": list(glob), "mae_global": list(glob.values())}).assign(
        mae_local=lambda f: f["series"].map(local)
    )
    table["gain"] = 1.0 - table["mae_global"] / table["mae_local"]
    table["better"] = np.where(table["gain"] > 0, "global", "local")
    return table
