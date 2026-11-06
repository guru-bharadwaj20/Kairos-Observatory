"""Acceleration-run detector.

On ``ln y`` a constant slope is exponential growth. A run of **consecutive slope
increases** means the growth rate itself keeps rising, which is what a
super-exponential (hyperbolic) path looks like when approximated piecewise. Such
runs are *candidates* for the singularity detectors of phase 5, not verdicts.

Only significant changepoints (``|delta| >= min_magnitude``) take part, so the
near-zero deltas left by Prophet's sparse prior neither start nor break a run.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from kairos.models.changepoints import CHANGEPOINT_COLUMNS

__all__ = ["RUN_COLUMNS", "acceleration_runs"]

RUN_COLUMNS: Final = (
    "series_id",
    "start",
    "end",
    "n_changepoints",
    "total_delta",
    "rate_start",
    "rate_end",
    "rate_ratio",
    "ongoing",
)


def _runs_one(
    rows: pd.DataFrame, min_run: int, last_cp: pd.Timestamp | None
) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    current: list[int] = []

    def close() -> None:
        if len(current) >= min_run:
            block = rows.iloc[current]
            rate_start = float(block["rate_before"].iloc[0])
            rate_end = float(block["rate_after"].iloc[-1])
            end = block["ds"].iloc[-1]
            runs.append(
                {
                    "series_id": block["series_id"].iloc[0],
                    "start": block["ds"].iloc[0],
                    "end": end,
                    "n_changepoints": len(current),
                    "total_delta": float(block["delta"].sum()),
                    "rate_start": rate_start,
                    "rate_end": rate_end,
                    "rate_ratio": rate_end / rate_start if rate_start > 0 else float("nan"),
                    "ongoing": last_cp is not None and end == last_cp,
                }
            )

    for i, delta in enumerate(rows["delta"].to_numpy()):
        if delta > 0:
            current.append(i)
        else:
            close()
            current = []
    close()
    return runs


def acceleration_runs(changepoints: pd.DataFrame, *, min_run: int = 2) -> pd.DataFrame:
    """Flag runs of consecutive positive slope deltas.

    Args:
        changepoints: Output of :func:`kairos.models.changepoints.changepoint_table`.
        min_run: Fewest consecutive significant increases that count as a run.

    Returns:
        One row per run with :data:`RUN_COLUMNS`. ``rate_ratio`` is the growth rate
        at the end over the rate at the start (``nan`` when the start rate is not
        positive); ``ongoing`` marks runs that end at the last significant
        changepoint of the series, i.e. acceleration still in progress.
    """
    if min_run < 1:
        raise ValueError("min_run must be >= 1")
    missing = [c for c in CHANGEPOINT_COLUMNS if c not in changepoints.columns]
    if missing:
        raise ValueError(f"not a changepoint table, missing {missing}")
    sig = changepoints.loc[changepoints["significant"]].sort_values(["series_id", "ds"])
    runs: list[dict[str, object]] = []
    for _, rows in sig.groupby("series_id", sort=False):
        rows = rows.reset_index(drop=True)
        last = rows["ds"].iloc[-1] if len(rows) else None
        runs.extend(_runs_one(rows, min_run, last))
    return pd.DataFrame(runs, columns=list(RUN_COLUMNS))
