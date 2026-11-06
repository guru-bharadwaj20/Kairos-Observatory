import math
import os
import sys

import cloudpickle
import pytest

from kairos.compute.parallel import pmap, resolve_n_jobs, worker_environment
from kairos.compute.threads import probe_blas_threads

# Test modules are not importable in spawned workers; ship these helpers by value.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def square(x):
    return x * x


def pid_and_env(_):
    return os.getpid(), os.environ.get("OMP_NUM_THREADS"), os.environ.get("KAIROS_RUN_ID")


def boom(x):
    if x == 3:
        raise ValueError("bad item 3")
    return x


def test_resolve_n_jobs(monkeypatch):
    monkeypatch.setenv("KAIROS_N_JOBS", "5")
    assert resolve_n_jobs(None) == 5
    assert resolve_n_jobs(8, n_tasks=3) == 3
    assert resolve_n_jobs(-1) == os.cpu_count()
    assert resolve_n_jobs(-2) == max(1, os.cpu_count() - 1)
    with pytest.raises(ValueError, match="0"):
        resolve_n_jobs(0)


def test_default_is_28(monkeypatch):
    monkeypatch.delenv("KAIROS_N_JOBS", raising=False)
    monkeypatch.delenv("KAIROS_CONFIG", raising=False)
    assert resolve_n_jobs(None, n_tasks=1000) == 28


def test_empty():
    assert pmap(square, [], n_jobs=2) == []


@pytest.mark.parametrize("backend", ["loky", "threading", "sequential"])
def test_ordered_results(backend):
    items = list(range(50))
    assert pmap(square, items, n_jobs=3, backend=backend) == [x * x for x in items]


def test_builtin_and_progress():
    assert pmap(math.sqrt, [1.0, 4.0, 9.0], n_jobs=2, desc="sqrt") == [1.0, 2.0, 3.0]


def test_loky_workers_are_separate_pinned_processes(monkeypatch):
    monkeypatch.setenv("KAIROS_RUN_ID", "run-pmap")
    out = pmap(pid_and_env, range(40), n_jobs=2)
    pids = {pid for pid, _, _ in out}
    assert os.getpid() not in pids
    assert {omp for _, omp, _ in out} == {"1"}
    assert {rid for _, _, rid in out} == {"run-pmap"}


def test_loky_workers_have_single_threaded_blas():
    for pools in pmap(probe_blas_threads, range(4), n_jobs=2):
        assert all(p["num_threads"] == 1 for p in pools), pools


def test_errors_propagate():
    with pytest.raises(ValueError, match="bad item 3"):
        pmap(boom, range(6), n_jobs=2)


def test_worker_environment_has_cmdstan():
    env = worker_environment()
    assert env["STAN_NUM_THREADS"] == "1"
    assert "CMDSTAN" in env
