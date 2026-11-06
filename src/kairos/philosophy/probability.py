"""``P(danger)`` for each dial from its sampling distribution.

The Jonas rule acts on a probability, so a dial must deliver a *distribution*, not
a point estimate. Dial producers put their bootstrap or ensemble draws of the dial's
danger statistic in ``DialReading.detail["samples"]``; ``P(danger)`` is the share of
those draws on the danger side of the dial's published threshold (``policy.yaml``,
section ``danger``). A Jeffreys interval reports how precise that share is.

A reading without draws is refused by default. ``allow_normal_fallback=True``
approximates the distribution as a normal fitted to the reported interval
(``detail["ci_level"]``, default 0.90); the result is labelled ``"normal-ci"`` so a
reader can see it was not taken from draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from kairos.types import DialReading

__all__ = [
    "SAMPLES_KEY",
    "DangerRule",
    "MissingDistributionError",
    "PDanger",
    "danger_rules",
    "p_danger",
    "p_danger_from_samples",
]

SAMPLES_KEY = "samples"
CI_LEVEL_KEY = "ci_level"
DEFAULT_CI_LEVEL = 0.90

Method = Literal["samples", "normal-ci"]


class MissingDistributionError(ValueError):
    """A reading carries no draws and the normal fallback is not allowed."""


class DangerRule(BaseModel):
    """When a dial's statistic is on the danger side.

    Attributes:
        direction: ``"above"`` (danger when ``x > threshold``) or ``"below"``.
        threshold: Boundary of the danger zone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: Literal["above", "below"]
    threshold: float

    def is_danger(self, x: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Element-wise danger test."""
        return x > self.threshold if self.direction == "above" else x < self.threshold


class PDanger(BaseModel):
    """Estimated probability that a dial is in its danger zone.

    Attributes:
        dial: Dial name.
        p: ``P(danger)``.
        lo: Lower end of the 95% Jeffreys interval for ``p``.
        hi: Upper end of that interval.
        n: Number of draws (0 for ``"normal-ci"``).
        method: ``"samples"`` or ``"normal-ci"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dial: str
    p: float = Field(ge=0, le=1)
    lo: float = Field(ge=0, le=1)
    hi: float = Field(ge=0, le=1)
    n: int = Field(ge=0)
    method: Method


def danger_rules(section: Mapping[str, Any]) -> dict[str, DangerRule]:
    """Parse the ``danger`` section of ``policy.yaml``."""
    return {dial: DangerRule.model_validate(raw) for dial, raw in section.items()}


def _clean(samples: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(samples, dtype=np.float64).ravel()
    finite: NDArray[np.float64] = x[np.isfinite(x)]
    if finite.size == 0:
        raise MissingDistributionError("no finite samples")
    return finite


def p_danger_from_samples(dial: str, samples: ArrayLike, rule: DangerRule) -> PDanger:
    """``P(danger)`` as the share of ``samples`` in the danger zone."""
    x = _clean(samples)
    k = int(rule.is_danger(x).sum())
    n = int(x.size)
    beta = stats.beta(k + 0.5, n - k + 0.5)
    lo = 0.0 if k == 0 else float(beta.ppf(0.025))
    hi = 1.0 if k == n else float(beta.ppf(0.975))
    return PDanger(dial=dial, p=k / n, lo=lo, hi=hi, n=n, method="samples")


def _normal_fallback(reading: DialReading, rule: DangerRule) -> PDanger:
    level = float(reading.detail.get(CI_LEVEL_KEY, DEFAULT_CI_LEVEL))
    if not 0 < level < 1:
        raise ValueError(f"ci_level must be in (0, 1), got {level}")
    z = float(stats.norm.ppf(0.5 + level / 2))
    sd = (reading.ci_hi - reading.ci_lo) / (2 * z)
    if sd <= 0:
        p_above = 1.0 if reading.value > rule.threshold else 0.0
    else:
        p_above = float(stats.norm.sf(rule.threshold, loc=reading.value, scale=sd))
    p = p_above if rule.direction == "above" else 1.0 - p_above
    return PDanger(dial=reading.dial, p=p, lo=p, hi=p, n=0, method="normal-ci")


def p_danger(
    reading: DialReading,
    rule: DangerRule,
    *,
    allow_normal_fallback: bool = False,
) -> PDanger:
    """``P(danger)`` for one reading.

    Raises:
        MissingDistributionError: If the reading has no draws and the fallback is off.
    """
    samples: Sequence[float] | None = reading.detail.get(SAMPLES_KEY)
    if samples is not None and len(samples) > 0:
        return p_danger_from_samples(reading.dial, samples, rule)
    if not allow_normal_fallback:
        raise MissingDistributionError(
            f"dial {reading.dial!r} has no detail[{SAMPLES_KEY!r}]; "
            "P(danger) needs bootstrap or ensemble draws, not a point estimate"
        )
    return _normal_fallback(reading, rule)
