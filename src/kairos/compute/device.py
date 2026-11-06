"""Device selection and precision policy.

* :func:`get_device` returns CUDA when asked for and available, else the CPU.
* Precision policy (CONTRIBUTING section 4.2): GPU kernels run in **float32**
  (Ada FP64 is ~1/64 of FP32); CPU reference implementations run in **float64**.
  :func:`dtype_for` encodes that rule.
* :func:`vram_budget_bytes` and :func:`chunk_size` keep peak VRAM under the
  configured budget (default 20 GB) so the desktop stays responsive.

torch is imported lazily so importing :mod:`kairos.compute` stays cheap.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["chunk_size", "cuda_available", "dtype_for", "get_device", "vram_budget_bytes"]


def cuda_available() -> bool:
    """Whether torch can use a CUDA device."""
    import torch

    return bool(torch.cuda.is_available())


def get_device(prefer: str | None = None) -> torch.device:
    """Return the compute device to use.

    Args:
        prefer: ``"auto"`` (CUDA if available), ``"cpu"``, ``"cuda"`` or ``"cuda:<n>"``.
            ``None`` uses ``Settings.device`` (``KAIROS_DEVICE``).

    Returns:
        A :class:`torch.device`. Asking for CUDA on a machine without it falls back
        to the CPU with a :class:`RuntimeWarning` instead of failing.

    Raises:
        ValueError: If ``prefer`` is not a recognised device string, or names a
            CUDA index that does not exist.
    """
    import torch

    if prefer is None:
        from kairos.settings import get_settings

        prefer = get_settings().device
    choice = prefer.strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "auto":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if choice == "cuda" or (choice.startswith("cuda:") and choice[5:].isdigit()):
        if not torch.cuda.is_available():
            warnings.warn(
                f"device {prefer!r} requested but CUDA is not available; using CPU",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        index = int(choice[5:]) if ":" in choice else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        return torch.device(f"cuda:{index}")
    raise ValueError(f"unknown device {prefer!r}; use auto, cpu, cuda or cuda:<n>")


def dtype_for(device: torch.device | str) -> torch.dtype:
    """Return the working precision for ``device``: float32 on GPU, float64 on CPU."""
    import torch

    kind = torch.device(device).type
    return torch.float32 if kind == "cuda" else torch.float64


def vram_budget_bytes(device: torch.device | str | None = None) -> int:
    """Bytes of VRAM a job may plan to use on ``device``.

    The budget is ``Settings.vram_budget_gb`` (default 20 GB), capped at 90 % of the
    device's total memory. Returns 0 for CPU devices.
    """
    import torch

    from kairos.settings import get_settings

    dev = get_device() if device is None else torch.device(device)
    if dev.type != "cuda":
        return 0
    total = int(torch.cuda.get_device_properties(dev).total_memory)
    return min(get_settings().vram_budget_bytes, int(total * 0.9))


def chunk_size(bytes_per_item: int, n_items: int, device: torch.device | str | None = None) -> int:
    """Largest number of items per chunk whose working set fits the VRAM budget.

    Args:
        bytes_per_item: Peak bytes one item needs (inputs, intermediates, outputs).
        n_items: Total items; the result never exceeds it.
        device: Target device; on CPU the whole job is one chunk.

    Returns:
        A chunk size in ``[1, n_items]`` (``0`` when ``n_items`` is 0).
    """
    if bytes_per_item <= 0:
        raise ValueError("bytes_per_item must be positive")
    if n_items <= 0:
        return 0
    budget = vram_budget_bytes(device)
    if budget == 0:
        return n_items
    return max(1, min(n_items, budget // bytes_per_item))
