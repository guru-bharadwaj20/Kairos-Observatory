"""Shared pytest configuration for the KAIROS test suite.

Markers:
    slow: long-running tests (backtests, large bootstraps); excluded in CI.
    gpu: tests that need a CUDA device; skipped automatically without one.
"""

from __future__ import annotations

import functools

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running test, excluded from the fast CI suite")
    config.addinivalue_line(
        "markers", "gpu: needs a CUDA device; auto-skipped when none is present"
    )


@functools.cache
def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if gpu_items and not _cuda_available():
        skip = pytest.mark.skip(reason="CUDA device not available")
        for item in gpu_items:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_settings():
    """Drop the cached Settings before and after each test so env changes apply."""
    from kairos.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
