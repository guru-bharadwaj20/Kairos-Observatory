"""Reproducibility: one call seeds every random number generator KAIROS touches.

:func:`seed_everything` covers Python's :mod:`random`, NumPy (legacy global state
and a returned :class:`numpy.random.Generator`), torch on CPU and CUDA, PyTorch
Lightning (used by NeuralProphet) and CmdStan, whose seed is not global and must
be passed to each fit: use :attr:`SeedState.stan_seed` or :func:`stan_seed`.

With ``deterministic=True`` (default), torch is switched to deterministic
kernels, so the same seed reproduces CPU results bit for bit. GPU results are
reproducible run to run on the same hardware and driver, but not across devices.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import random
import sys
from dataclasses import dataclass

import numpy as np

__all__ = ["SeedState", "derive_seed", "seed_everything", "stan_seed"]

_MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class SeedState:
    """What :func:`seed_everything` set.

    Attributes:
        seed: The master seed.
        rng: A fresh NumPy generator seeded from ``seed``; prefer it to the global state.
        stan_seed: Seed to pass to CmdStan / Prophet fits (``Prophet().fit(df, seed=...)``).
        torch_seeded: Whether torch was imported and seeded.
        lightning_seeded: Whether PyTorch Lightning was found and seeded.
    """

    seed: int
    rng: np.random.Generator
    stan_seed: int
    torch_seeded: bool
    lightning_seeded: bool


def derive_seed(seed: int, *keys: object) -> int:
    """Derive a stable child seed from ``seed`` and ``keys`` (e.g. a series ID and fold).

    Uses SHA-256, so it is identical across processes, platforms and Python
    versions, unlike :func:`hash`. Useful for giving each pooled task its own seed.
    """
    text = "\x1f".join([str(seed), *map(repr, keys)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def stan_seed(seed: int | None = None) -> int:
    """Return the CmdStan seed for ``seed`` (default ``Settings.seed``)."""
    if seed is None:
        from kairos.settings import get_settings

        seed = get_settings().seed
    return derive_seed(seed, "cmdstan")


def seed_everything(seed: int | None = None, *, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy, torch (CPU and CUDA), Lightning and CmdStan.

    Args:
        seed: Master seed in ``[0, 2**32)``; ``None`` uses ``Settings.seed``.
        deterministic: Ask torch for deterministic algorithms (warn-only for ops
            without one) and disable cuDNN autotuning.

    Returns:
        A :class:`SeedState` with the generator and per-library seeds.
    """
    if seed is None:
        from kairos.settings import get_settings

        seed = get_settings().seed
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)  # inherited by spawned workers
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global state used by third-party code
    rng = np.random.default_rng(seed)

    torch_seeded = _seed_torch(seed, deterministic)
    lightning_seeded = _seed_lightning(seed)
    return SeedState(seed, rng, stan_seed(seed), torch_seeded, lightning_seeded)


def _seed_torch(seed: int, deterministic: bool) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return False
    if deterministic:
        # Required by cuBLAS for deterministic results; must be set before CUDA work.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return True


def _seed_lightning(seed: int) -> bool:
    for name in ("lightning.pytorch", "pytorch_lightning", "lightning_fabric"):
        if name not in sys.modules and importlib.util.find_spec(name.split(".")[0]) is None:
            continue
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        seed_fn = getattr(module, "seed_everything", None)
        if callable(seed_fn):
            _quiet_lightning()
            seed_fn(seed, workers=True)
            return True
    return False


def _quiet_lightning() -> None:
    import logging

    for name in ("lightning.fabric.utilities.seed", "lightning_fabric.utilities.seed"):
        logging.getLogger(name).setLevel(logging.WARNING)
