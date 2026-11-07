"""Löb independence report: no dial may be certified by the lab it measures.

By Löb's theorem a system cannot be the final judge of its own soundness; in
practice, no dial may rest on one lab-controlled source alone. Each reading lists
its provenance in ``detail["sources"]`` as ``[{"source", "source_group",
"lab_controlled"}]``. A dial **fails** when

* it lists no provenance at all (independence cannot be shown), or
* all of its sources fall in a single source group and that group is lab-controlled
  (flagged on a source, or named in ``tripwires.yaml: independence``).

:func:`enforce` raises :class:`LobIndependenceError`, which fails the run.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from kairos.logging import get_logger
from kairos.tripwires.aggregator import Independence
from kairos.types import DialReading

__all__ = [
    "DialIndependence",
    "IndependenceReport",
    "LobIndependenceError",
    "SourceProvenance",
    "enforce",
    "independence_report",
]

log = get_logger(__name__)


class LobIndependenceError(RuntimeError):
    """A dial depends only on a single lab-controlled source."""


class SourceProvenance(BaseModel):
    """Where one input of a dial comes from."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    source: str
    source_group: str
    lab_controlled: bool = False


class DialIndependence(BaseModel):
    """Independence verdict for one dial.

    Attributes:
        dial: Dial name.
        groups: Distinct source groups.
        lab_controlled_groups: Those groups that are lab-controlled.
        passed: Whether the dial is independent enough.
        reason: Why it failed (empty when passed).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dial: str
    groups: list[str]
    lab_controlled_groups: list[str]
    passed: bool
    reason: str = ""


class IndependenceReport(BaseModel):
    """Independence verdicts for every reading of a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dials: list[DialIndependence] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every dial passed."""
        return all(d.passed for d in self.dials)

    @property
    def failures(self) -> list[DialIndependence]:
        """Dials that failed."""
        return [d for d in self.dials if not d.passed]


def _check(reading: DialReading, cfg: Independence) -> DialIndependence:
    raw = reading.detail.get(cfg.sources_key) or []
    sources = [SourceProvenance.model_validate(s) for s in raw]
    groups = sorted({s.source_group for s in sources})
    lab = sorted(
        {s.source_group for s in sources if s.lab_controlled}
        | (set(groups) & set(cfg.lab_controlled_groups))
    )
    if not sources:
        reason = f"no provenance in detail[{cfg.sources_key!r}]"
    elif len(groups) == 1 and lab:
        reason = f"depends only on lab-controlled source group {groups[0]!r}"
    else:
        reason = ""
    return DialIndependence(
        dial=reading.dial,
        groups=groups,
        lab_controlled_groups=lab,
        passed=not reason,
        reason=reason,
    )


def independence_report(
    readings: Iterable[DialReading], config: Independence | None = None
) -> IndependenceReport:
    """Check every reading's provenance."""
    cfg = config or Independence()
    return IndependenceReport(dials=[_check(r, cfg) for r in readings])


def enforce(report: IndependenceReport) -> IndependenceReport:
    """Return ``report`` if it passed.

    Raises:
        LobIndependenceError: If any dial failed.
    """
    if not report.passed:
        msg = "; ".join(f"{d.dial}: {d.reason}" for d in report.failures)
        log.error("Löb independence check failed: %s", msg)
        raise LobIndependenceError(msg)
    return report
