"""Parallel map for the many small fits KAIROS runs (Prophet, bootstraps, sweeps).

:func:`pmap` wraps :class:`joblib.Parallel` on the ``loky`` backend:

* **Default 28 workers** from :class:`kairos.settings.Settings` (``KAIROS_N_JOBS``).
* **Pinned workers:** every native thread pool runs single-threaded
  (:mod:`kairos.compute.threads`), so 28 workers never oversubscribe 32 threads.
* **Dynamic scheduling:** ``batch_size="auto"`` with small pre-dispatch, so fast
  P-cores keep pulling work while slower E-cores finish theirs.
* **Windows-spawn safe:** loky always starts fresh interpreters. ``fn`` must be a
  module-level function (or a picklable callable); lambdas and closures fail.
* **Ordered:** results come back in input order.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any, Literal

from joblib import Parallel, delayed, parallel_config

from kairos.compute.threads import limit_blas_threads, pin_worker_env

__all__ = ["Backend", "pmap", "resolve_n_jobs", "worker_environment"]

Backend = Literal["loky", "threading", "sequential"]

_applied_key: tuple[int, int, frozenset[tuple[str, str]]] | None = None


def resolve_n_jobs(n_jobs: int | None, n_tasks: int | None = None) -> int:
    """Turn a user ``n_jobs`` into a concrete worker count.

    Args:
        n_jobs: ``None`` uses ``Settings.n_jobs``; ``-1`` uses every logical CPU;
            other negative values follow joblib (``-2`` = all but one).
        n_tasks: Never start more workers than tasks.

    Returns:
        A worker count ``>= 1``.
    """
    if n_jobs is None:
        from kairos.settings import get_settings

        n_jobs = get_settings().n_jobs
    cpus = os.cpu_count() or 1
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0")
    workers = n_jobs if n_jobs > 0 else max(1, cpus + 1 + n_jobs)
    if n_tasks is not None:
        workers = min(workers, max(1, n_tasks))
    return max(1, workers)


def worker_environment(threads: int = 1) -> dict[str, str]:
    """Environment exported to each worker: thread pins, run ID and CmdStan location."""
    from kairos.compute.threads import single_thread_env
    from kairos.logging import get_run_id

    env = single_thread_env(threads)
    env["KAIROS_RUN_ID"] = get_run_id()
    if "CMDSTAN" in os.environ:
        env["CMDSTAN"] = os.environ["CMDSTAN"]
    else:
        from kairos.compute.hardware import stan_info

        info = stan_info()
        if info.ok and info.cmdstan_path:
            env["CMDSTAN"] = info.cmdstan_path
    return env


class _PinnedCall:
    """Picklable wrapper that pins the worker's threads once, then calls ``fn``."""

    __slots__ = ("env", "fn", "threads")

    def __init__(self, fn: Callable[[Any], Any], threads: int, env: dict[str, str]) -> None:
        self.fn = fn
        self.threads = threads
        self.env = env

    def __call__(self, item: Any) -> Any:
        global _applied_key
        # loky reuses workers across pmap calls; re-pin only when the settings change.
        key = (os.getpid(), self.threads, frozenset(self.env.items()))
        if _applied_key != key:
            pin_worker_env(self.threads, self.env)
            _applied_key = key
        return self.fn(item)


def pmap[T, R](
    fn: Callable[[T], R],
    iterable: Iterable[T],
    n_jobs: int | None = None,
    desc: str | None = None,
    backend: Backend = "loky",
    *,
    threads_per_worker: int = 1,
    batch_size: int | Literal["auto"] = "auto",
) -> list[R]:
    """Apply ``fn`` to every item in parallel and return the results in order.

    Args:
        fn: Module-level function of one argument (must be picklable for ``loky``).
        iterable: Inputs; materialised into a list first.
        n_jobs: Worker count (see :func:`resolve_n_jobs`). ``1`` runs in-process.
        desc: If given, show a progress bar with this label.
        backend: ``"loky"`` (processes, default), ``"threading"`` for GIL-releasing
            work, or ``"sequential"`` for debugging.
        threads_per_worker: Native threads per worker (BLAS, OpenMP, Stan, torch).
        batch_size: joblib batch size; ``"auto"`` adapts to task duration.

    Returns:
        ``[fn(x) for x in iterable]``, computed in parallel.

    Raises:
        Exception: The first exception raised by ``fn`` is re-raised in the caller.
    """
    items = list(iterable)
    if not items:
        return []
    workers = resolve_n_jobs(n_jobs, len(items))
    if backend == "sequential" or workers == 1:
        backend = "sequential"

    if backend == "sequential":
        with limit_blas_threads(threads_per_worker):
            return _with_progress((fn(x) for x in items), len(items), desc)

    if backend == "threading":
        with limit_blas_threads(threads_per_worker), parallel_config(backend="threading"):
            gen = Parallel(n_jobs=workers, batch_size=batch_size, return_as="generator")(
                delayed(fn)(x) for x in items
            )
            return _with_progress(gen, len(items), desc)

    call = _PinnedCall(fn, threads_per_worker, worker_environment(threads_per_worker))
    with parallel_config(backend="loky", inner_max_num_threads=threads_per_worker):
        gen = Parallel(
            n_jobs=workers,
            batch_size=batch_size,
            pre_dispatch="2*n_jobs",
            return_as="generator",
        )(delayed(call)(x) for x in items)
        return _with_progress(gen, len(items), desc)


def _with_progress[R](results: Iterable[R], total: int, desc: str | None) -> list[R]:
    if desc is None:
        return list(results)
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    out: list[R] = []
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(desc, total=total)
        for result in results:
            out.append(result)
            progress.advance(task)
    return out
