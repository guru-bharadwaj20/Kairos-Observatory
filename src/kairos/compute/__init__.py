"""Compute layer: thread governance, parallel map, device selection, seeds and fit cache."""

from kairos.compute.device import chunk_size, dtype_for, get_device, vram_budget_bytes
from kairos.compute.parallel import pmap, resolve_n_jobs
from kairos.compute.seeds import SeedState, derive_seed, seed_everything, stan_seed
from kairos.compute.threads import (
    THREAD_ENV_VARS,
    blas_thread_counts,
    limit_blas_threads,
    pin_worker_env,
    single_thread_env,
)

__all__ = [
    "THREAD_ENV_VARS",
    "SeedState",
    "blas_thread_counts",
    "chunk_size",
    "derive_seed",
    "dtype_for",
    "get_device",
    "limit_blas_threads",
    "pin_worker_env",
    "pmap",
    "resolve_n_jobs",
    "seed_everything",
    "single_thread_env",
    "stan_seed",
    "vram_budget_bytes",
]
