"""Compute layer: thread governance, parallel map, device selection, seeds and fit cache."""

from kairos.compute.parallel import pmap, resolve_n_jobs
from kairos.compute.threads import (
    THREAD_ENV_VARS,
    blas_thread_counts,
    limit_blas_threads,
    pin_worker_env,
    single_thread_env,
)

__all__ = [
    "THREAD_ENV_VARS",
    "blas_thread_counts",
    "limit_blas_threads",
    "pin_worker_env",
    "pmap",
    "resolve_n_jobs",
    "single_thread_env",
]
