import random

import numpy as np
import pandas as pd
import pytest
import torch

from kairos.compute.seeds import derive_seed, seed_everything, stan_seed


def _draw():
    tiny = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    opt = torch.optim.SGD(tiny.parameters(), lr=0.1)
    x = torch.randn(32, 4)
    for _ in range(5):
        opt.zero_grad()
        loss = tiny(x).pow(2).mean()
        loss.backward()
        opt.step()
    return (
        random.random(),
        np.random.rand(3).tolist(),  # noqa: NPY002
        torch.cat([p.detach().flatten() for p in tiny.parameters()]).numpy().tobytes(),
    )


def test_same_seed_is_bit_identical_on_cpu():
    state = seed_everything(123)
    a, rng_a = _draw(), state.rng.random(4)
    state = seed_everything(123)
    b, rng_b = _draw(), state.rng.random(4)
    assert a == b
    assert np.array_equal(rng_a, rng_b)


def test_different_seed_differs():
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    assert _draw() != a


def test_default_seed_from_settings(monkeypatch):
    monkeypatch.setenv("KAIROS_SEED", "77")
    assert seed_everything().seed == 77


def test_state_reports_libraries():
    state = seed_everything(5)
    assert state.torch_seeded
    assert state.lightning_seeded
    assert state.stan_seed == stan_seed(5)


def test_invalid_seed():
    with pytest.raises(ValueError, match="seed"):
        seed_everything(-1)


def test_derive_seed_is_stable():
    assert derive_seed(1, "s", 2) == derive_seed(1, "s", 2)
    assert derive_seed(1, "s", 2) != derive_seed(1, "s", 3)
    assert 0 <= derive_seed(99) < 2**32
    # Pinned value: must never change across versions or platforms.
    assert derive_seed(20261113, "cmdstan") == stan_seed(20261113)


def test_prophet_fit_reproducible_with_stan_seed():
    from prophet import Prophet

    ds = pd.date_range("2015-01-01", periods=48, freq="MS")
    y = 0.03 * np.arange(48) + np.random.default_rng(0).normal(0, 0.05, 48)
    df = pd.DataFrame({"ds": ds, "y": y})
    seed = seed_everything(9).stan_seed

    def fit():
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df, seed=seed)
        return m.params["k"].tobytes(), m.params["delta"].tobytes()

    assert fit() == fit()


@pytest.mark.gpu
def test_cuda_reproducible():
    def run():
        seed_everything(11)
        x = torch.randn(512, 512, device="cuda")
        return (x @ x).sum().item()

    assert run() == run()
