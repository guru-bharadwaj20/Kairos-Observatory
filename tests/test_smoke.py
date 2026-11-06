import importlib
import pkgutil

import pytest

import kairos

SUBPACKAGES = [
    "compute",
    "data",
    "data.sources",
    "models",
    "neural",
    "accel",
    "vinge",
    "jonas",
    "tripwires",
    "philosophy",
    "validation",
    "ledger",
    "dashboard",
    "commands",
]


def test_version_is_set():
    assert isinstance(kairos.__version__, str)
    assert kairos.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable_with_docstring(name):
    module = importlib.import_module(f"kairos.{name}")
    assert module.__doc__


def test_no_unexpected_top_level_import_errors():
    for info in pkgutil.iter_modules(kairos.__path__):
        importlib.import_module(f"kairos.{info.name}")


@pytest.mark.gpu
def test_cuda_visible():
    import torch

    assert torch.cuda.is_available()
    assert torch.zeros(1, device="cuda").item() == 0.0
