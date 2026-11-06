import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import pytest
from threadpoolctl import threadpool_info

from kairos.compute.threads import (
    THREAD_ENV_VARS,
    blas_thread_counts,
    limit_blas_threads,
    pin_worker_env,
    probe_blas_threads,
    single_thread_env,
)

MANY_CORES = (os.cpu_count() or 1) >= 4


def test_single_thread_env_covers_all_vars():
    env = single_thread_env()
    assert set(env) == set(THREAD_ENV_VARS)
    assert set(env.values()) == {"1"}
    assert "STAN_NUM_THREADS" in env
    with pytest.raises(ValueError, match="threads"):
        single_thread_env(0)


def test_limit_blas_threads_context():
    probe_blas_threads()  # make sure a BLAS is loaded
    with limit_blas_threads(1):
        assert all(p["num_threads"] == 1 for p in blas_thread_counts())


def _pool(initializer=None, initargs=()):
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(2, mp_context=ctx, initializer=initializer, initargs=initargs)


def test_pinned_spawn_workers_are_single_threaded():
    """Workers initialised with pin_worker_env run every native pool with 1 thread."""
    with _pool(pin_worker_env) as pool:
        reports = list(pool.map(probe_blas_threads, range(4)))
    for pools in reports:
        assert pools, "no BLAS thread pool detected in the worker"
        assert all(p["num_threads"] == 1 for p in pools), pools


@pytest.mark.skipif(not MANY_CORES, reason="needs >= 4 cores to observe oversubscription")
def test_unpinned_workers_would_oversubscribe(monkeypatch):
    """Control: without pinning, BLAS in a worker grabs several threads."""
    for var in THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    with _pool() as pool:
        pools = pool.submit(probe_blas_threads).result()
    assert max(p["num_threads"] for p in pools) > 1


def test_pin_worker_env_in_process(monkeypatch):
    for var in THREAD_ENV_VARS:
        monkeypatch.setenv(var, "8")
    original = threadpool_info()
    try:
        pin_worker_env(environ={"CMDSTAN_TEST": "1"})
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["CMDSTAN_TEST"] == "1"
        assert all(p["num_threads"] == 1 for p in blas_thread_counts())
    finally:
        from threadpoolctl import threadpool_limits

        for p in original:
            threadpool_limits(limits=p["num_threads"], user_api=p["user_api"])
