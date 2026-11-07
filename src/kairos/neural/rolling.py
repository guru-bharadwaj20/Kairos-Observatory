"""Rolling ``κ(t)`` and the test for "rising".

``κ`` is recomputed on trailing windows of ``window`` months that end every
``step`` months. Each window gets its own hold-out (the last
``spec.test_months`` of the window), seeds and permutation null. All fits of
all windows are flattened into one batch for the CPU pool.

"Rising" (idea doc §8) is tested on the window estimates with the one-sided
**Mann–Kendall** test (Kendall's ``τ`` against time) and the **Theil–Sen**
slope with its confidence interval. ``κ(t)`` is rising when the Mann–Kendall
p-value is below ``alpha`` and the slope is positive. Overlapping windows make
the estimates serially correlated, which inflates significance; the default
``step`` therefore equals ``spec.test_months`` so hold-out windows don't
overlap, and the docs flag the remaining caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from kairos.neural.ensemble import DEFAULT_N_SEEDS, run_tasks
from kairos.neural.kappa import (
    DEFAULT_N_PERMUTATIONS,
    KappaResult,
    kappa_from_results,
    kappa_reading,
    kappa_tasks,
)
from kairos.neural.paired import PairedSpec
from kairos.types import DialReading

__all__ = [
    "TrendTest",
    "rolling_kappa",
    "rolling_reading",
    "rolling_table",
    "rolling_windows",
    "trend_test",
]


@dataclass(frozen=True)
class TrendTest:
    """Mann–Kendall and Theil–Sen trend test of a sequence.

    Attributes:
        tau: Kendall's ``τ`` of the values against time.
        p_value: One-sided p-value for an increasing trend.
        slope: Theil–Sen slope per step.
        slope_lo: Lower bound of the slope interval.
        slope_hi: Upper bound of the slope interval.
        n: Number of points.
        alpha: Significance level.
    """

    tau: float
    p_value: float
    slope: float
    slope_lo: float
    slope_hi: float
    n: int
    alpha: float = 0.05

    @property
    def rising(self) -> bool:
        """Significant increasing trend with a positive slope."""
        return bool(self.p_value < self.alpha and self.slope > 0)


def trend_test(values: np.ndarray | list[float], *, alpha: float = 0.05) -> TrendTest:  # type: ignore[type-arg]
    """Test ``values`` (in time order) for an increasing trend.

    Raises:
        ValueError: With fewer than 3 finite values.
    """
    y = np.asarray(values, dtype=np.float64)
    t = np.arange(len(y), dtype=np.float64)
    keep = np.isfinite(y)
    y, t = y[keep], t[keep]
    if len(y) < 3:
        raise ValueError("need at least 3 finite values for a trend test")
    tau, p = stats.kendalltau(t, y, alternative="greater")
    slope, _, lo, hi = stats.theilslopes(y, t, alpha=1.0 - 2.0 * alpha)
    tau = 0.0 if not np.isfinite(tau) else float(tau)
    p = 1.0 if not np.isfinite(p) else float(p)
    return TrendTest(tau, p, float(slope), float(lo), float(hi), len(y), alpha)


def rolling_windows(
    design: pd.DataFrame, window: int, step: int
) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Trailing windows of ``window`` rows ending every ``step`` rows, latest included.

    Raises:
        ValueError: If ``window`` exceeds the data or ``step < 1``.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    df = design.sort_values("ds").reset_index(drop=True)
    if window > len(df):
        raise ValueError(f"window {window} longer than the data ({len(df)} rows)")
    ends = list(range(len(df), window - 1, -step))[::-1]
    return [(pd.Timestamp(df["ds"].iloc[e - 1]), df.iloc[e - window : e].copy()) for e in ends]


def rolling_kappa(
    design: pd.DataFrame,
    spec: PairedSpec | None = None,
    *,
    window: int = 120,
    step: int | None = None,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int | None = None,
    device: str | None = None,
    n_jobs: int | None = None,
    use_cache: bool = True,
) -> list[KappaResult]:
    """``κ`` on every rolling window, in time order (``key`` = window end ``YYYY-MM-DD``)."""
    spec = spec or PairedSpec()
    step = spec.test_months if step is None else step
    windows = rolling_windows(design, window, step)
    tasks = []
    for end, part in windows:
        tasks += kappa_tasks(
            part,
            spec,
            n_seeds=n_seeds,
            n_permutations=n_permutations,
            seed=seed,
            device=device,
            key=end.date().isoformat(),
        )
    results = run_tasks(tasks, n_jobs=n_jobs, use_cache=use_cache, desc="rolling κ")
    out = []
    for end, _ in windows:
        key = end.date().isoformat()
        out.append(kappa_from_results([r for r in results if r.key == key], key=key))
    return out


def rolling_table(results: list[KappaResult]) -> pd.DataFrame:
    """One row per window: end, kappa, interval, p-value and excess over the null."""
    return pd.DataFrame(
        {
            "window_end": pd.to_datetime([r.key for r in results]),
            "kappa": [r.kappa for r in results],
            "ci_lo": [r.ci_lo for r in results],
            "ci_hi": [r.ci_hi for r in results],
            "p_value": [r.p_value for r in results],
            "excess": [r.excess for r in results],
        }
    )


def rolling_reading(
    results: list[KappaResult], as_of: date | None = None, *, alpha: float = 0.05
) -> DialReading:
    """Recursion dial from rolling results: latest ``κ``, trend test in ``detail``.

    The level is 1 when the latest ``κ`` beats its null and ``κ(t)`` is rising.
    """
    if not results:
        raise ValueError("no rolling results")
    latest = results[-1]
    trend = trend_test([r.kappa for r in results], alpha=alpha) if len(results) >= 3 else None
    as_of = as_of or pd.Timestamp(latest.key).date()
    reading = kappa_reading(latest, as_of, rising=bool(trend and trend.rising))
    detail = dict(reading.detail)
    detail["trend"] = None if trend is None else {**trend.__dict__, "rising": trend.rising}
    detail["rolling"] = (
        rolling_table(results)
        .assign(window_end=lambda f: f["window_end"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )
    return reading.model_copy(update={"detail": detail})
