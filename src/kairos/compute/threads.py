"""Thread governance: stop BLAS/OpenMP oversubscription in worker pools.

With 28 worker processes on a 32-thread CPU, every native thread pool inside a
worker (OpenBLAS/MKL, OpenMP, numexpr, torch intra-op, Stan) must run with one
thread, or the machine thrashes (CONTRIBUTING section 4.1).

Two layers are used:

* :data:`THREAD_ENV_VARS` are exported *before* a worker imports numpy, so the
  native libraries start single-threaded (:func:`pin_worker_env`).
* :func:`limit_blas_threads` uses :mod:`threadpoolctl` to clamp libraries that are
  already loaded, for code that runs in the parent process.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from typing import Any

from threadpoolctl import threadpool_info, threadpool_limits

__all__ = [
    "THREAD_ENV_VARS",
    "blas_thread_counts",
    "limit_blas_threads",
    "pin_worker_env",
    "probe_blas_threads",
    "single_thread_env",
]

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "STAN_NUM_THREADS",
)
"""Environment variables that size native thread pools (including CmdStan's)."""


def single_thread_env(threads: int = 1) -> dict[str, str]:
    """Return the environment mapping that pins every native pool to ``threads``."""
    if threads < 1:
        raise ValueError("threads must be >= 1")
    return dict.fromkeys(THREAD_ENV_VARS, str(threads))


def pin_worker_env(threads: int = 1, environ: Mapping[str, str] | None = None) -> None:
    """Pin native thread pools in the *current* process.

    Call this first thing in a worker. It sets the environment variables (which
    govern libraries loaded later), clamps already-loaded BLAS/OpenMP pools with
    threadpoolctl, and sets torch's intra-op threads if torch is already imported.

    Args:
        threads: Threads per native pool.
        environ: Extra variables to export as well (e.g. CmdStan settings).
    """
    os.environ.update(single_thread_env(threads))
    if environ:
        os.environ.update(environ)
    threadpool_limits(limits=threads)
    torch = sys.modules.get("torch")
    if torch is not None:
        # torch may refuse once parallel work has started; the env vars still apply.
        with suppress(RuntimeError):
            torch.set_num_threads(threads)


@contextmanager
def limit_blas_threads(limit: int = 1) -> Iterator[None]:
    """Clamp BLAS and OpenMP pools of loaded libraries to ``limit`` threads inside the block.

    Example:
        >>> import numpy as np
        >>> with limit_blas_threads():
        ...     _ = np.ones((4, 4)) @ np.ones((4, 4))
    """
    with threadpool_limits(limits=limit):
        yield


def blas_thread_counts() -> list[dict[str, Any]]:
    """Report each loaded native thread pool as ``{"api", "library", "num_threads"}``."""
    return [
        {
            "api": pool.get("user_api"),
            "library": pool.get("internal_api"),
            "num_threads": pool.get("num_threads"),
        }
        for pool in threadpool_info()
    ]


def probe_blas_threads(_: object = None) -> list[dict[str, Any]]:
    """Load numpy and SciPy's BLAS, run a matmul and report the thread pools.

    A top-level function so it can be sent to spawned workers; used by tests and
    ``kairos doctor``-style diagnostics to prove workers are pinned.
    """
    import numpy as np
    import scipy.linalg

    a = np.ones((64, 64))
    scipy.linalg.blas.dgemm(1.0, a, a)
    return blas_thread_counts()
