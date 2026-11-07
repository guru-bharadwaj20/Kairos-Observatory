"""Seed ensembles: many small NeuralProphet fits, mean and spread of skill.

With ~200 points one fit is noisy, so every ``κ`` uses at least
:data:`DEFAULT_N_SEEDS` (20) seeds per model (CONTRIBUTING D9).

Execution policy (``docs/benchmarks.md``): a single small fit takes 0.64 s on
the CPU and 1.21 s on the GPU, so ensembles run as **independent CPU fits over
the** :func:`kairos.compute.pmap` **pool** (28 single-threaded workers). The GPU
is used only when asked for explicitly (``device="cuda"``); fits then run one
after another in-process, because CUDA contexts must not be forked into many
workers. :func:`ensemble_device` encodes the rule.

Results are memoised with :func:`kairos.compute.cached_fit`, so rerunning an
ensemble on unchanged data and settings costs nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from kairos.compute import cached_fit, derive_seed, pmap
from kairos.logging import get_logger
from kairos.neural.paired import FitResult, FitTask, PairedSpec, fit_task, paired_tasks

__all__ = [
    "DEFAULT_N_SEEDS",
    "ensemble_device",
    "ensemble_seeds",
    "run_seed_ensemble",
    "run_tasks",
    "summarize",
]

DEFAULT_N_SEEDS = 20
log = get_logger(__name__)


@cached_fit(version="1")
def _cached_fit_task(task: FitTask) -> FitResult:
    return fit_task(task)


def _run_cached(task: FitTask) -> FitResult:
    return _cached_fit_task(task)


def ensemble_device(device: str | None = None) -> str:
    """Device for ensemble fits: CPU unless CUDA is requested explicitly.

    ``None`` and ``"auto"`` mean the CPU pool (faster for small fits); ``"cuda"``
    or ``"cuda:<n>"`` are honoured when CUDA is available, else CPU.
    """
    if device is None or device.strip().lower() in {"auto", "cpu"}:
        return "cpu"
    from kairos.compute import get_device

    return str(get_device(device))


def ensemble_seeds(seed: int, n: int, *keys: object) -> list[int]:
    """``n`` stable child seeds of ``seed`` (different ``keys`` give independent sets)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [derive_seed(seed, "ensemble", *keys, i) for i in range(n)]


def run_tasks(
    tasks: Sequence[FitTask],
    *,
    n_jobs: int | None = None,
    use_cache: bool = True,
    desc: str | None = None,
) -> list[FitResult]:
    """Run fit tasks, in order, following the execution policy.

    CPU tasks go to the ``pmap`` pool; any CUDA task makes the whole batch run
    sequentially in-process.
    """
    fn = _run_cached if use_cache else fit_task
    if any(t.device.startswith("cuda") for t in tasks):
        log.info("running %d NeuralProphet fits in-process on the GPU", len(tasks))
        return [fn(t) for t in tasks]
    return pmap(fn, list(tasks), n_jobs=n_jobs, desc=desc)


def run_seed_ensemble(
    design: pd.DataFrame,
    spec: PairedSpec,
    *,
    n_seeds: int = DEFAULT_N_SEEDS,
    seed: int | None = None,
    device: str | None = None,
    n_jobs: int | None = None,
    use_cache: bool = True,
) -> list[FitResult]:
    """Fit the paired ``exo``/``self`` models over ``n_seeds`` seeds."""
    if seed is None:
        from kairos.settings import get_settings

        seed = get_settings().seed
    seeds = ensemble_seeds(seed, n_seeds)
    tasks = paired_tasks(design, spec, seeds, device=ensemble_device(device))
    return run_tasks(tasks, n_jobs=n_jobs, use_cache=use_cache)


def summarize(results: Sequence[FitResult]) -> pd.DataFrame:
    """Mean and spread of out-of-sample MAE per variant.

    Returns:
        One row per variant: ``n``, ``mean``, ``std``, ``q05``, ``median``, ``q95``.
    """
    frame = pd.DataFrame({"variant": [r.variant for r in results], "mae": [r.mae for r in results]})
    grouped = frame.groupby("variant")["mae"]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean": grouped.mean(),
            "std": grouped.std(ddof=1),
            "q05": grouped.quantile(0.05),
            "median": grouped.median(),
            "q95": grouped.quantile(0.95),
        }
    )
    return out.replace({np.nan: 0.0}).reset_index()
