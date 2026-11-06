"""Hardware and toolchain probes used by ``kairos doctor`` and the benchmarks.

Every probe is cheap, side-effect free and never raises: failures are reported
as values (``None`` or ``ok=False``) so callers can decide what is fatal.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import importlib.resources
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil

__all__ = [
    "CpuInfo",
    "CudaInfo",
    "StanInfo",
    "cpu_info",
    "cuda_info",
    "package_version",
    "ram_bytes",
    "stan_info",
]


@dataclass(frozen=True)
class CpuInfo:
    """CPU topology.

    Attributes:
        model: Processor brand string.
        physical_cores: Number of physical cores.
        logical_cores: Number of hardware threads.
        performance_cores: P-core count on hybrid CPUs, ``None`` if unknown.
        efficiency_cores: E-core count on hybrid CPUs, ``None`` if unknown.
    """

    model: str
    physical_cores: int
    logical_cores: int
    performance_cores: int | None = None
    efficiency_cores: int | None = None

    @property
    def is_hybrid(self) -> bool:
        """Whether the CPU mixes performance and efficiency cores."""
        return bool(self.performance_cores and self.efficiency_cores)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {**asdict(self), "is_hybrid": self.is_hybrid}


@dataclass(frozen=True)
class CudaInfo:
    """What torch can see of the CUDA stack.

    Attributes:
        torch_version: ``torch.__version__`` or ``None`` if torch is not importable.
        torch_cuda_build: CUDA version torch was built against (``None`` for CPU builds).
        available: ``torch.cuda.is_available()``.
        devices: One dict per device with ``name``, ``total_bytes`` and ``capability``.
        error: Import or runtime error message, if any.
    """

    torch_version: str | None
    torch_cuda_build: str | None
    available: bool
    devices: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class StanInfo:
    """Status of the CmdStan toolchain that Prophet uses.

    Attributes:
        prophet_version: Installed prophet version, ``None`` if missing.
        cmdstan_path: Bundled or configured CmdStan directory.
        model_file: Prophet's compiled Stan model executable.
        ok: True when the compiled model exists and CmdStan is located.
        error: Explanation when ``ok`` is false.
    """

    prophet_version: str | None
    cmdstan_path: str | None
    model_file: str | None
    ok: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


def package_version(name: str) -> str | None:
    """Return the installed version of distribution ``name`` or ``None``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def ram_bytes() -> int:
    """Total physical memory in bytes."""
    return int(psutil.virtual_memory().total)


def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _hybrid_split_windows() -> tuple[int, int] | None:
    """Count cores per efficiency class with ``GetLogicalProcessorInformationEx``.

    On Intel hybrid CPUs, P-cores report the highest ``EfficiencyClass``.
    Returns ``(performance, efficiency)`` or ``None`` if the CPU is not hybrid.
    """
    relation_processor_core = 0
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009 (Windows-only attribute)
    length = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(length))
    if length.value == 0:
        return None
    buffer = ctypes.create_string_buffer(length.value)
    if not kernel32.GetLogicalProcessorInformationEx(
        relation_processor_core, buffer, ctypes.byref(length)
    ):
        return None
    classes: list[int] = []
    offset = 0
    raw = buffer.raw
    while offset < length.value:
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship (DWORD), Size (DWORD),
        # then PROCESSOR_RELATIONSHIP: Flags (BYTE), EfficiencyClass (BYTE), ...
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        classes.append(raw[offset + 9])
        if size == 0:
            break
        offset += size
    if not classes or len(set(classes)) < 2:
        return None
    top = max(classes)
    performance = sum(1 for c in classes if c == top)
    return performance, len(classes) - performance


def _hybrid_split_linux() -> tuple[int, int] | None:
    """Read the ``cpu_core``/``cpu_atom`` PMU cpu lists exposed on Intel hybrid CPUs."""
    base = Path("/sys/devices")
    core_list, atom_list = base / "cpu_core" / "cpus", base / "cpu_atom" / "cpus"
    if not (core_list.exists() and atom_list.exists()):
        return None

    def threads(path: Path) -> set[int]:
        out: set[int] = set()
        for part in path.read_text().strip().split(","):
            if "-" in part:
                lo, hi = part.split("-")
                out.update(range(int(lo), int(hi) + 1))
            elif part:
                out.add(int(part))
        return out

    def cores(cpus: set[int]) -> int:
        ids = set()
        for cpu in cpus:
            topo = base / "system" / "cpu" / f"cpu{cpu}" / "topology" / "core_id"
            ids.add(topo.read_text().strip() if topo.exists() else str(cpu))
        return len(ids)

    try:
        return cores(threads(core_list)), cores(threads(atom_list))
    except (OSError, ValueError):
        return None


def cpu_info() -> CpuInfo:
    """Probe the CPU model, core counts and the hybrid P/E split."""
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    split: tuple[int, int] | None = None
    try:
        if sys.platform == "win32":
            split = _hybrid_split_windows()
        elif sys.platform.startswith("linux"):
            split = _hybrid_split_linux()
    except Exception:  # probing must never crash the doctor
        split = None
    return CpuInfo(
        model=_cpu_model(),
        physical_cores=int(physical),
        logical_cores=int(logical),
        performance_cores=split[0] if split else None,
        efficiency_cores=split[1] if split else None,
    )


def cuda_info() -> CudaInfo:
    """Report the torch build and the CUDA devices it can use."""
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return CudaInfo(None, None, False, error=f"torch import failed: {exc}")
    version = str(torch.__version__)
    build = torch.version.cuda
    try:
        available = bool(torch.cuda.is_available())
        devices = []
        for index in range(torch.cuda.device_count() if available else 0):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_bytes": int(props.total_memory),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    except Exception as exc:
        return CudaInfo(version, build, False, error=f"CUDA query failed: {exc}")
    return CudaInfo(version, build, available, devices)


def stan_info() -> StanInfo:
    """Locate Prophet's bundled CmdStan and compiled model without fitting anything."""
    version = package_version("prophet")
    if version is None:
        return StanInfo(None, None, None, False, "prophet is not installed")
    try:
        root = Path(str(importlib.resources.files("prophet"))) / "stan_model"
    except Exception as exc:
        return StanInfo(version, None, None, False, f"cannot locate prophet package: {exc}")
    exe = "prophet_model.bin"
    model_file = root / exe
    bundled = sorted(root.glob("cmdstan-*"))
    cmdstan = bundled[-1] if bundled else None
    if cmdstan is None:
        try:
            cmdstanpy = importlib.import_module("cmdstanpy")
            cmdstan = Path(cmdstanpy.cmdstan_path())
        except Exception:
            cmdstan = None
    if not model_file.exists():
        return StanInfo(
            version, str(cmdstan) if cmdstan else None, None, False, "compiled model missing"
        )
    if cmdstan is None:
        return StanInfo(version, None, str(model_file), False, "CmdStan not found")
    return StanInfo(version, str(cmdstan), str(model_file), True)
