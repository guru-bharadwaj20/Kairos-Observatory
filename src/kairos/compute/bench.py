"""Hardware micro-benchmarks behind ``scripts/bench_hardware.py`` (K-1.08).

Each function returns plain numbers so the results can be rendered to
``docs/benchmarks.md`` and later used to pick chunk sizes and CPU-vs-GPU defaults.
Worker functions are module-level so they pickle into spawned workers.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kairos.compute.parallel import pmap

__all__ = [
    "BenchResult",
    "bench_lstsq",
    "bench_matmul",
    "bench_neuralprophet",
    "bench_prophet",
    "prophet_fit_task",
    "synthetic_monthly",
]


@dataclass(frozen=True)
class BenchResult:
    """One benchmark measurement.

    Attributes:
        name: Benchmark and configuration.
        seconds: Wall-clock seconds for the whole run.
        work: Units of work done (fits, solves, FLOP).
        unit: Name of a unit of work.
        extra: Free-form details (sizes, devices).
    """

    name: str
    seconds: float
    work: float
    unit: str
    extra: dict[str, Any]

    @property
    def rate(self) -> float:
        """Units of work per second."""
        return self.work / self.seconds if self.seconds > 0 else float("inf")


def synthetic_monthly(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Log-scale monthly series with two growth regimes and noise (Prophet ``ds``/``y``)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    slope = np.where(t < n // 2, 0.02, 0.05)
    y = np.cumsum(slope) + rng.normal(0, 0.03, n)
    return pd.DataFrame({"ds": pd.date_range("2008-01-01", periods=n, freq="MS"), "y": y})


def _quiet() -> None:
    for name in ("cmdstanpy", "prophet", "NP", "pytorch_lightning", "lightning_fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).propagate = False
    warnings.filterwarnings("ignore")


def prophet_fit_task(seed: int) -> float:
    """Fit one Prophet null model on a synthetic 200-point series; return fit seconds."""
    _quiet()
    from prophet import Prophet

    df = synthetic_monthly(seed=seed)
    start = time.perf_counter()
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
    )
    model.fit(df, seed=seed)
    return time.perf_counter() - start


def bench_prophet(n_fits: int, n_jobs: int) -> BenchResult:
    """Time ``n_fits`` Prophet fits with ``n_jobs`` workers (1 = serial in-process)."""
    _quiet()
    if n_jobs > 1:  # warm the pool so process start-up is not counted
        pmap(prophet_fit_task, range(n_jobs), n_jobs=n_jobs)
    else:
        prophet_fit_task(0)
    start = time.perf_counter()
    per_fit = pmap(prophet_fit_task, range(n_fits), n_jobs=n_jobs)
    seconds = time.perf_counter() - start
    return BenchResult(
        f"prophet n_jobs={n_jobs}",
        seconds,
        n_fits,
        "fits",
        {"median_fit_s": float(np.median(per_fit)), "n_jobs": n_jobs},
    )


def bench_neuralprophet(accelerator: str, epochs: int = 50, n: int = 200) -> BenchResult:
    """Time one NeuralProphet AR fit (``n_lags=12``, ``n_forecasts=6``) on ``accelerator``.

    Args:
        accelerator: ``"cpu"`` or ``"gpu"``.
        epochs: Training epochs.
        n: Series length.
    """
    _quiet()
    from neuralprophet import NeuralProphet, set_log_level

    set_log_level("ERROR")
    df = synthetic_monthly(n=n)
    model = NeuralProphet(
        n_lags=12,
        n_forecasts=6,
        epochs=epochs,
        batch_size=32,
        learning_rate=0.01,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        accelerator=accelerator,
        trainer_config={"enable_progress_bar": False, "enable_model_summary": False},
    )
    start = time.perf_counter()
    model.fit(df, freq="MS", progress=None, minimal=True)
    seconds = time.perf_counter() - start
    return BenchResult(
        f"neuralprophet {accelerator}",
        seconds,
        epochs,
        "epochs",
        {"n": n, "epochs": epochs},
    )


def _lstsq_problem(batch: int, n: int, dtype: Any, device: Any, seed: int = 0) -> tuple[Any, Any]:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.linspace(-1, 1, n, dtype=torch.float64)
    phase = torch.rand(batch, 1, generator=gen, dtype=torch.float64) * 6.28
    freq = 2 + 10 * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
    design = torch.stack(
        [
            torch.ones(batch, n, dtype=torch.float64),
            t.expand(batch, n),
            torch.cos(freq * t + phase),
            torch.sin(freq * t + phase),
        ],
        dim=-1,
    )
    coef = torch.randn(batch, 4, 1, generator=gen, dtype=torch.float64)
    y = design @ coef + 0.01 * torch.randn(batch, n, 1, generator=gen, dtype=torch.float64)
    return design.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)


def _solve_normal_equations(design: Any, y: Any) -> Any:
    import torch

    # A contiguous transpose is ~15x faster than a strided one for tiny batched
    # matmuls on CUDA (cuBLAS picks a far better batched-GEMM kernel).
    design_t = design.transpose(-1, -2).contiguous()
    return torch.linalg.solve(design_t @ design, design_t @ y)


def bench_lstsq(batch: int, n: int = 200, device: str = "cpu", repeats: int = 3) -> BenchResult:
    """Batched 4-parameter least squares (the LPPLS inner solve) via normal equations.

    CPU runs in float64 (the reference precision); CUDA runs in float32.
    """
    import torch

    from kairos.compute.device import dtype_for

    dev = torch.device(device)
    dtype = dtype_for(dev)
    design, y = _lstsq_problem(batch, n, dtype, dev)
    _solve_normal_equations(design, y)  # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _solve_normal_equations(design, y)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return BenchResult(
        f"lstsq {dev.type} {str(dtype).removeprefix('torch.')}",
        seconds,
        batch * repeats,
        "solves",
        {"batch": batch, "n": n},
    )


def bench_matmul(dtype_name: str, size: int = 8192, repeats: int = 10) -> BenchResult:
    """Square GPU matmul throughput; ``work`` is in FLOP (``2 * size**3`` per product)."""
    import torch

    dtype = getattr(torch, dtype_name)
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    del a, b
    torch.cuda.empty_cache()
    return BenchResult(
        f"matmul cuda {dtype_name}", seconds, 2.0 * size**3 * repeats, "FLOP", {"size": size}
    )
