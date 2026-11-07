"""The recursion coefficient ``κ`` with a permutation null.

Definition (idea doc §8, CONTRIBUTING D9). With ``MAE`` the out-of-sample mean
absolute error (log units, averaged over horizons and seeds)::

    κ = 1 - MAE_self / MAE_exo

* ``κ = 0``: the target's own lags add nothing once compute, money and people
  are known; ``κ > 0``: they cut the forecast error by that fraction (``κ = 0.2``
  means 20 % lower error); ``κ < 0``: they hurt (overfitting).
* **Interval**: percentile bootstrap over the paired seeds (default 90 %).
* **Null**: ``n_permutations`` refits of the ``self`` model with a
  block-shuffled self channel give ``κ_null = 1 - MAE_null / MAE_exo``; the
  one-sided p-value is ``(1 + #{κ_null >= κ}) / (1 + n_permutations)``.
  ``excess = κ - median(κ_null)`` is the gain over chance.

:func:`kappa_reading` turns a result into the ``"recursion"``
:class:`~kairos.types.DialReading`. The danger sign is ``κ`` *rising* across
rolling windows (K-4.09), so a single reading only reaches level 1 when the
caller passes ``rising=True`` and ``κ`` beats its null.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from kairos.neural.ensemble import (
    DEFAULT_N_SEEDS,
    ensemble_device,
    ensemble_seeds,
    run_tasks,
)
from kairos.neural.paired import FitResult, FitTask, PairedSpec, paired_tasks
from kairos.types import DialReading

__all__ = [
    "DEFAULT_N_PERMUTATIONS",
    "DIAL",
    "KappaResult",
    "estimate_kappa",
    "kappa_from_results",
    "kappa_reading",
    "kappa_tasks",
]

DIAL = "recursion"
DEFAULT_N_PERMUTATIONS = 49
"""Null refits: with 49 the smallest attainable p-value is 0.02."""


@dataclass(frozen=True)
class KappaResult:
    """``κ`` with its interval and permutation null.

    Attributes:
        kappa: Ensemble ``κ = 1 - mean(MAE_self) / mean(MAE_exo)``.
        ci_lo: Lower bound of the seed-bootstrap interval.
        ci_hi: Upper bound of the seed-bootstrap interval.
        ci_level: Coverage of the interval.
        p_value: One-sided permutation p-value (``nan`` without permutations).
        null_kappas: ``κ`` of every null refit.
        per_seed: ``1 - MAE_self / MAE_exo`` of each seed pair.
        mae_exo: Mean MAE of the exogenous-only model.
        mae_self: Mean MAE of the self-lag model.
        alpha: Significance level used by :attr:`significant`.
        key: Label of the fits (e.g. a rolling-window end).
    """

    kappa: float
    ci_lo: float
    ci_hi: float
    ci_level: float
    p_value: float
    null_kappas: tuple[float, ...]
    per_seed: tuple[float, ...]
    mae_exo: float
    mae_self: float
    alpha: float = 0.05
    key: str = ""

    @property
    def excess(self) -> float:
        """``κ`` minus the median null ``κ`` (gain over chance)."""
        if not self.null_kappas:
            return float("nan")
        return self.kappa - float(np.median(self.null_kappas))

    @property
    def significant(self) -> bool:
        """Whether ``κ`` beats the permutation null at :attr:`alpha`."""
        return bool(np.isfinite(self.p_value) and self.p_value < self.alpha)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (null and per-seed values included)."""
        return {
            "kappa": self.kappa,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "ci_level": self.ci_level,
            "p_value": self.p_value,
            "excess": self.excess,
            "significant": self.significant,
            "mae_exo": self.mae_exo,
            "mae_self": self.mae_self,
            "n_seeds": len(self.per_seed),
            "n_permutations": len(self.null_kappas),
            "null_kappas": list(self.null_kappas),
            "per_seed": list(self.per_seed),
            "key": self.key,
        }


def kappa_tasks(
    design: pd.DataFrame,
    spec: PairedSpec,
    *,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int | None = None,
    device: str | None = None,
    key: str = "",
) -> list[FitTask]:
    """All fits behind one ``κ``: paired exo/self per seed plus the null refits."""
    if seed is None:
        from kairos.settings import get_settings

        seed = get_settings().seed
    dev = ensemble_device(device)
    seeds = ensemble_seeds(seed, n_seeds, key)
    tasks = paired_tasks(design, spec, seeds, device=dev, key=key)
    perm_seeds = ensemble_seeds(seed, max(n_permutations, 1), key, "null")
    base = FitTask(design, spec, "null", 0, device=dev, key=key)
    tasks += [
        replace(base, seed=seeds[j % n_seeds], perm=j, perm_seed=perm_seeds[j])
        for j in range(n_permutations)
    ]
    return tasks


def kappa_from_results(
    results: Sequence[FitResult],
    *,
    ci_level: float = 0.90,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    key: str = "",
) -> KappaResult:
    """Compute ``κ``, its bootstrap interval and permutation p-value from fits.

    Raises:
        ValueError: If no complete exo/self seed pair is present.
    """
    exo = {r.seed: r.mae for r in results if r.variant == "exo"}
    own = {r.seed: r.mae for r in results if r.variant == "self"}
    seeds = sorted(set(exo) & set(own))
    if not seeds:
        raise ValueError("need at least one paired exo/self result")
    e = np.array([exo[s] for s in seeds])
    o = np.array([own[s] for s in seeds])
    kappa = float(1.0 - o.mean() / e.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seeds), size=(n_boot, len(seeds)))
    boot = 1.0 - o[idx].mean(axis=1) / e[idx].mean(axis=1)
    tail = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(boot, [tail, 1.0 - tail])

    nulls = tuple(
        float(1.0 - r.mae / e.mean())
        for r in sorted((r for r in results if r.variant == "null"), key=lambda r: r.perm)
    )
    p = (1 + sum(k >= kappa for k in nulls)) / (1 + len(nulls)) if nulls else float("nan")
    return KappaResult(
        kappa=kappa,
        ci_lo=float(min(lo, kappa)),
        ci_hi=float(max(hi, kappa)),
        ci_level=ci_level,
        p_value=float(p),
        null_kappas=nulls,
        per_seed=tuple(float(v) for v in 1.0 - o / e),
        mae_exo=float(e.mean()),
        mae_self=float(o.mean()),
        alpha=alpha,
        key=key,
    )


def estimate_kappa(
    design: pd.DataFrame,
    spec: PairedSpec | None = None,
    *,
    n_seeds: int = DEFAULT_N_SEEDS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int | None = None,
    device: str | None = None,
    n_jobs: int | None = None,
    use_cache: bool = True,
    ci_level: float = 0.90,
    alpha: float = 0.05,
) -> KappaResult:
    """Estimate ``κ`` on a design frame from :func:`kairos.neural.regressors.design_frame`.

    Args:
        design: Wide frame ``ds, y, <driver columns>``.
        spec: Paired-model settings (default :class:`PairedSpec`).
        n_seeds: Seeds per model (>= 20 recommended).
        n_permutations: Null refits (0 disables the null).
        seed: Master seed (default ``Settings.seed``).
        device: ``None``/``"auto"``/``"cpu"`` = CPU pool; ``"cuda"`` = in-process GPU.
        n_jobs: Pool size (default ``Settings.n_jobs``).
        use_cache: Reuse cached fits.
        ci_level: Coverage of the seed-bootstrap interval.
        alpha: Significance level.
    """
    spec = spec or PairedSpec()
    tasks = kappa_tasks(
        design, spec, n_seeds=n_seeds, n_permutations=n_permutations, seed=seed, device=device
    )
    results = run_tasks(tasks, n_jobs=n_jobs, use_cache=use_cache)
    return kappa_from_results(results, ci_level=ci_level, alpha=alpha)


def kappa_reading(result: KappaResult, as_of: date, *, rising: bool = False) -> DialReading:
    """The ``"recursion"`` dial reading for ``result``.

    Args:
        result: Output of :func:`estimate_kappa` (usually the latest rolling window).
        as_of: Data vintage of the reading.
        rising: Whether the rolling trend test (K-4.09) found ``κ`` rising.

    Returns:
        ``value = κ`` with its bootstrap interval; ``level = 1`` (danger zone) only
        when ``κ`` is significant against its null **and** rising, else ``0``.
        ``detail`` carries the p-value, excess over the null and the MAEs.
    """
    level = 1 if (rising and result.significant) else 0
    detail = result.as_dict()
    detail["rising"] = rising
    return DialReading(
        dial=DIAL,
        value=result.kappa,
        level=level,
        ci_lo=result.ci_lo,
        ci_hi=result.ci_hi,
        as_of=as_of,
        detail=detail,
    )
