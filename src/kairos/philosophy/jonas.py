"""Jonas's heuristic of fear as an asymmetric loss (design note D6).

Let a missed catastrophe cost ``λ`` and a false alarm cost ``1``. Raising the alarm
has expected loss ``1 − P`` and staying silent has expected loss ``λ·P``, so the
alarm is the better act exactly when ``λ·P > 1 − P``, that is when
``P(danger) > 1 / (1 + λ)``.

``λ`` is the one number philosophy must supply. It is read from the public,
versioned ``policy.yaml`` shipped next to this module (or from an explicit path),
never hard-coded.
"""

from __future__ import annotations

from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kairos.logging import get_logger
from kairos.tripwires.state import Hysteresis

__all__ = [
    "DEFAULT_POLICY",
    "JonasPolicy",
    "PolicyChange",
    "PolicySpec",
    "alarm_threshold",
    "load_policy",
]

log = get_logger(__name__)

DEFAULT_POLICY = "policy.yaml"


def alarm_threshold(lam: float) -> float:
    """Return ``1 / (1 + λ)``, the probability above which the alarm is raised."""
    if not lam > 0:
        raise ValueError(f"lambda must be > 0, got {lam}")
    return 1.0 / (1.0 + lam)


class JonasPolicy(BaseModel):
    """The Jonas decision rule.

    Attributes:
        lam: Loss ratio ``λ`` = cost of a missed catastrophe / cost of a false alarm.
        band: Hysteresis width below the alarm threshold.
        dwell: Consecutive readings needed for a dial to change state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    lam: float = Field(alias="lambda", gt=0)
    band: float = Field(default=0.0, ge=0, lt=1)
    dwell: int = Field(default=1, ge=1)

    @property
    def threshold(self) -> float:
        """``1 / (1 + λ)``."""
        return alarm_threshold(self.lam)

    def alarm(self, p_danger: float) -> bool:
        """Whether the alarm should be raised (strict ``λ·P > 1 − P``)."""
        if not 0.0 <= p_danger <= 1.0:
            raise ValueError(f"P(danger) must be in [0, 1], got {p_danger}")
        return self.lam * p_danger > 1.0 - p_danger

    def expected_losses(self, p_danger: float) -> dict[str, float]:
        """Expected loss of each act, in units of one false alarm."""
        return {"alarm": 1.0 - p_danger, "silent": self.lam * p_danger}

    def hysteresis(self) -> Hysteresis:
        """Hysteresis band whose upper edge is the alarm threshold."""
        return Hysteresis.around(self.threshold, min(self.band, self.threshold), self.dwell)


class PolicyChange(BaseModel):
    """One changelog entry of the policy file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    date: date
    note: str


class PolicySpec(BaseModel):
    """Contents of ``policy.yaml``.

    Attributes:
        version: Semantic version of the policy.
        changelog: Why each version exists.
        jonas: The Jonas rule.
        extra: Other sections (read by the modules that own them).
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    version: str
    changelog: list[PolicyChange] = Field(default_factory=list)
    jonas: JonasPolicy

    def section(self, name: str) -> dict[str, Any]:
        """Return a raw extra section, or ``{}`` if absent."""
        raw = (self.model_extra or {}).get(name, {})
        if not isinstance(raw, dict):
            raise ValueError(f"policy section {name!r} must be a mapping")
        return raw


def read_yaml(path: Path | None, package: str, default: str) -> dict[str, Any]:
    """Read a YAML mapping from ``path`` or from package data ``package/default``."""
    if path is None:
        text = resources.files(package).joinpath(default).read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("policy file must contain a YAML mapping")
    return data


def load_policy(path: Path | None = None) -> PolicySpec:
    """Load the public policy file (default: the packaged ``policy.yaml``)."""
    spec = PolicySpec.model_validate(read_yaml(path, "kairos.philosophy", DEFAULT_POLICY))
    log.debug("policy %s: lambda=%s", spec.version, spec.jonas.lam)
    return spec
