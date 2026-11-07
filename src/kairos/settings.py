"""Runtime configuration for KAIROS.

Values are resolved in this order (first wins):

1. keyword arguments passed to :class:`Settings`;
2. environment variables with the ``KAIROS_`` prefix (``KAIROS_N_JOBS=8``);
3. a YAML file: ``$KAIROS_CONFIG`` if set, else ``./kairos.yaml`` if it exists;
4. the defaults below, tuned for the reference workstation (docs/design.md, hardware plan).

Policy parameters such as ``λ`` and tripwire thresholds are **not** settings; they
live in versioned YAML under ``config/`` (Jonas rule).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

__all__ = ["DEFAULT_SEED", "Settings", "config_file", "get_settings"]

DEFAULT_SEED = 20261113
"""Default seed: von Foerster's doomsday, Friday 13 November 2026."""

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def config_file() -> Path | None:
    """Return the YAML config file to read, or ``None`` if there is none."""
    explicit = os.environ.get("KAIROS_CONFIG")
    if explicit:
        return Path(explicit)
    local = Path("kairos.yaml")
    return local if local.is_file() else None


class Settings(BaseSettings):
    """KAIROS runtime settings.

    Attributes:
        data_dir: Root of the dataset tree (``raw/``, ``processed/``, vintages).
        cache_dir: Fit cache and other disposable artefacts.
        n_jobs: Worker processes for CPU pools; 28 leaves 4 of 32 threads free.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:<n>"``.
        vram_budget_gb: Peak VRAM the GPU kernels may plan for.
        seed: Global seed used by :func:`kairos.compute.seed_everything`.
        log_level: Root level for ``kairos.*`` loggers.
        log_format: ``"rich"`` for terminals, ``"json"`` for one JSON object per line.
        run_id: Identifier stamped on every log line; generated when not given.
    """

    model_config = SettingsConfigDict(
        env_prefix="KAIROS_",
        extra="ignore",
        validate_default=True,
    )

    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache/kairos")
    n_jobs: int = Field(default=28, ge=-1)
    device: str = "auto"
    vram_budget_gb: float = Field(default=20.0, gt=0)
    seed: int = Field(default=DEFAULT_SEED, ge=0, lt=2**32)
    log_level: LogLevel = "INFO"
    log_format: Literal["rich", "json"] = "rich"
    run_id: str | None = None

    @field_validator("n_jobs")
    @classmethod
    def _no_zero_jobs(cls, value: int) -> int:
        if value == 0:
            raise ValueError("n_jobs must be positive or -1 (all cores)")
        return value

    @field_validator("device")
    @classmethod
    def _valid_device(cls, value: str) -> str:
        v = value.strip().lower()
        if v in {"auto", "cpu", "cuda"} or (v.startswith("cuda:") and v[5:].isdigit()):
            return v
        raise ValueError(f"device must be auto, cpu, cuda or cuda:<n>, got {value!r}")

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML file between environment variables and defaults."""
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        path = config_file()
        if path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=path))
        return tuple(sources)

    @property
    def vram_budget_bytes(self) -> int:
        """VRAM budget in bytes."""
        return int(self.vram_budget_gb * 1024**3)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached; call ``get_settings.cache_clear()`` to reload)."""
    return Settings()
