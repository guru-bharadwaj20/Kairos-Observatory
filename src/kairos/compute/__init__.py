"""Compute layer: thread governance, parallel map, device selection, seeds and fit cache."""

from kairos.compute.cache import CachedFit, cached_fit, clear_fit_cache, data_hash, get_fit_cache
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
    "CachedFit",
    "SeedState",
    "blas_thread_counts",
    "cached_fit",
    "chunk_size",
    "clear_fit_cache",
    "data_hash",
    "derive_seed",
    "dtype_for",
    "get_device",
    "get_fit_cache",
    "limit_blas_threads",
    "pin_worker_env",
    "pmap",
    "resolve_n_jobs",
    "seed_everything",
    "single_thread_env",
    "stan_seed",
    "vram_budget_bytes",
]


def __getattr__(name: str) -> object:
    # ``kairos.compute.fit_cache`` is the default joblib.Memory, resolved lazily.
    if name == "fit_cache":
        return get_fit_cache()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
