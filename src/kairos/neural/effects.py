"""Event effect table: "which policies bend the curve".

For every event a model with drivers and events is fitted over a seed
ensemble. The effect of an event in one fit is NeuralProphet's
``event_<name>`` component at each occurrence month (averaged over
occurrences), in log units; ``peak`` is the largest absolute value inside the
window. Across seeds the table reports the mean and a percentile interval, and
``factor = exp(effect)`` gives the multiplicative change in raw units
(``0.9`` = 10 % lower). The interval reflects training noise, not sampling
uncertainty of the data; an interval excluding 0 is necessary, not sufficient,
evidence of an effect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kairos.compute import pmap
from kairos.neural.ensemble import DEFAULT_N_SEEDS, ensemble_seeds
from kairos.neural.events import EventSpec, add_events, with_events
from kairos.neural.model import NeuralConfig, build_model, fit_model, predict
from kairos.neural.regressors import DriverSpec, add_lagged_drivers

__all__ = ["DEFAULT_EFFECT_CONFIG", "event_effect_table", "event_effects"]

DEFAULT_EFFECT_CONFIG = NeuralConfig(
    epochs=300, batch_size=16, learning_rate=0.1, n_changepoints=0, early_stopping=False
)


@dataclass(frozen=True)
class _EffectTask:
    design: pd.DataFrame
    events: tuple[EventSpec, ...]
    drivers: tuple[DriverSpec, ...]
    config: NeuralConfig
    seed: int


def event_effects(task: _EffectTask) -> dict[str, tuple[float, float]]:
    """Fit once and return ``{event: (effect at occurrence, peak in window)}``."""
    cols = ["ds", "y", *(d.column for d in task.drivers)]
    model = build_model(task.config, "cpu")
    add_lagged_drivers(model, task.drivers)
    add_events(model, task.events)
    df = with_events(model, task.design[cols].reset_index(drop=True), task.events)
    fit_model(model, df, config=task.config, seed=task.seed)
    forecast = predict(model, df).set_index("ds")
    out = {}
    for spec in task.events:
        comp = forecast[f"event_{spec.name}"].astype(np.float64)
        on_date = [float(comp.loc[d]) for d in spec.dates if d in comp.index]
        nonzero = comp[comp != 0]
        peak = float(nonzero.loc[nonzero.abs().idxmax()]) if len(nonzero) else 0.0
        out[spec.name] = (float(np.mean(on_date)) if on_date else float("nan"), peak)
    return out


def event_effect_table(
    design: pd.DataFrame,
    events: Sequence[EventSpec],
    *,
    drivers: Sequence[DriverSpec] = (),
    config: NeuralConfig = DEFAULT_EFFECT_CONFIG,
    n_seeds: int = DEFAULT_N_SEEDS,
    seed: int = 0,
    ci_level: float = 0.90,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Estimated effect and interval for each event.

    Returns:
        One row per event: ``event``, ``effect`` (mean, log units), ``ci_lo``,
        ``ci_hi``, ``peak``, ``factor``, ``factor_lo``, ``factor_hi``,
        ``excludes_zero`` and ``n_seeds``.
    """
    if not events:
        raise ValueError("no events given")
    tasks = [
        _EffectTask(design, tuple(events), tuple(drivers), config, s)
        for s in ensemble_seeds(seed, n_seeds, "effects")
    ]
    runs = pmap(event_effects, tasks, n_jobs=n_jobs)
    tail = (1.0 - ci_level) / 2.0
    records = []
    for spec in events:
        vals = np.array([r[spec.name][0] for r in runs])
        peaks = np.array([r[spec.name][1] for r in runs])
        lo, hi = np.nanquantile(vals, [tail, 1.0 - tail])
        mean = float(np.nanmean(vals))
        records.append(
            {
                "event": spec.name,
                "effect": mean,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "peak": float(np.nanmean(peaks)),
                "factor": float(np.exp(mean)),
                "factor_lo": float(np.exp(lo)),
                "factor_hi": float(np.exp(hi)),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "n_seeds": len(vals),
            }
        )
    return pd.DataFrame.from_records(records)
