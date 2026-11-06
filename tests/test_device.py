import pytest
import torch

from kairos.compute import device as dev_mod
from kairos.compute.device import chunk_size, dtype_for, get_device, vram_budget_bytes
from kairos.settings import get_settings


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_cpu_explicit():
    assert get_device("cpu") == torch.device("cpu")


def test_auto_without_cuda(no_cuda):
    assert get_device("auto").type == "cpu"


def test_cuda_request_falls_back_with_warning(no_cuda):
    with pytest.warns(RuntimeWarning, match="CUDA is not available"):
        assert get_device("cuda").type == "cpu"


def test_settings_device_used_by_default(monkeypatch):
    monkeypatch.setenv("KAIROS_DEVICE", "cpu")
    assert get_device().type == "cpu"


def test_unknown_device():
    with pytest.raises(ValueError, match="unknown device"):
        get_device("tpu")


def test_precision_policy():
    assert dtype_for("cpu") == torch.float64
    assert dtype_for(torch.device("cuda:0")) == torch.float32


def test_cpu_budget_and_chunks():
    assert vram_budget_bytes("cpu") == 0
    assert chunk_size(1024, 500, device="cpu") == 500
    assert chunk_size(1024, 0, device="cpu") == 0
    with pytest.raises(ValueError, match="positive"):
        chunk_size(0, 10, device="cpu")


@pytest.mark.gpu
def test_auto_picks_cuda():
    assert get_device("auto") == torch.device("cuda:0")
    with pytest.raises(ValueError, match="does not exist"):
        get_device(f"cuda:{torch.cuda.device_count()}")


@pytest.mark.gpu
def test_vram_budget_is_capped(monkeypatch):
    budget = vram_budget_bytes("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory
    assert 0 < budget <= min(20 * 1024**3, 0.9 * total)
    monkeypatch.setenv("KAIROS_VRAM_BUDGET_GB", "1")
    get_settings.cache_clear()
    assert vram_budget_bytes("cuda:0") == 1024**3
    assert chunk_size(1024**2, 10_000, device="cuda:0") == 1024


@pytest.mark.gpu
def test_float32_kernel_matches_float64_reference():
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(256, 256, generator=gen, dtype=torch.float64)
    ref = a @ a.T
    gpu = (a.to("cuda", dtype_for("cuda")) @ a.to("cuda", dtype_for("cuda")).T).cpu().double()
    assert torch.allclose(gpu, ref, rtol=1e-3, atol=1e-3)
    assert dev_mod.cuda_available()
