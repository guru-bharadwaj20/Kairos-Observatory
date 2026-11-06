import pytest

from kairos.compute.bench import (
    BenchResult,
    bench_lstsq,
    bench_matmul,
    bench_prophet,
    prophet_fit_task,
    synthetic_monthly,
)


def test_synthetic_monthly_shape():
    df = synthetic_monthly(n=24)
    assert list(df.columns) == ["ds", "y"]
    assert len(df) == 24
    assert df["ds"].dt.is_month_start.all()


def test_rate():
    assert BenchResult("x", 2.0, 10, "fits", {}).rate == 5.0


def test_lstsq_cpu_is_correct_and_float64():
    import torch

    from kairos.compute.bench import _lstsq_problem, _solve_normal_equations

    design, y = _lstsq_problem(8, 50, torch.float64, torch.device("cpu"))
    coef = _solve_normal_equations(design, y)
    ref = torch.linalg.lstsq(design, y).solution
    assert torch.allclose(coef, ref, atol=1e-8)
    r = bench_lstsq(100, n=50, device="cpu", repeats=1)
    assert r.name == "lstsq cpu float64"
    assert r.work == 100


def test_prophet_task_runs():
    assert prophet_fit_task(0) > 0


@pytest.mark.slow
def test_bench_prophet_pool():
    r = bench_prophet(n_fits=4, n_jobs=2)
    assert r.work == 4
    assert r.rate > 0


@pytest.mark.gpu
def test_gpu_benchmarks():
    r = bench_lstsq(1000, n=50, device="cuda", repeats=1)
    assert r.name == "lstsq cuda float32"
    m = bench_matmul("float32", size=512, repeats=1)
    assert m.rate > 0
