"""The Jonas dial: ``dG/dt``, its interval, run length and state (K-7.06).

Parameters come from the published ``jonas.yaml`` shipped with this package (or a
file passed explicitly), never from literals in code.

States, from least to most alarming:

=================== ==========================================================
``closing``          latest ``dG/dt <= 0``
``widening``         latest ``dG/dt > 0``
``persistent``       run-length rule met (``min_run`` positive months)
``sustained``        run-length rule **and** the HAC t-test both met
``sustained_robust`` sustained, and the whole reported ``dG/dt`` interval is > 0
=================== ==========================================================

Each state maps to a tripwire contribution via ``levels`` in the YAML.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Final, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kairos.jonas.attribution import component_attribution
from kairos.jonas.scenarios import closing_scenarios
from kairos.jonas.sustained import SustainedResult, sustained_positive
from kairos.jonas.velocity import gap_velocity
from kairos.types import DialReading, SeriesFrame

__all__ = [
    "DIAL_NAME",
    "STATES",
    "JonasParams",
    "JonasState",
    "classify",
    "jonas_dial",
    "load_params",
]

DIAL_NAME: Final = "jonas"
JonasState = Literal["closing", "widening", "persistent", "sustained", "sustained_robust"]
STATES: Final[tuple[JonasState, ...]] = (
    "closing",
    "widening",
    "persistent",
    "sustained",
    "sustained_robust",
)


class JonasParams(BaseModel):
    """Published parameters of the Jonas dial (see ``jonas.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    base_year: int
    window: int = Field(ge=3)
    n_boot: int = Field(ge=0)
    level: float = Field(gt=0, lt=1)
    min_run: int = Field(ge=1)
    alpha: float = Field(gt=0, lt=1)
    test_months: int = Field(ge=3)
    scenario_months: int = Field(ge=3)
    levels: dict[JonasState, int]

    @model_validator(mode="after")
    def _all_levels(self) -> JonasParams:
        missing = [s for s in STATES if s not in self.levels]
        if missing:
            raise ValueError(f"levels missing states: {missing}")
        if any(not 0 <= v <= 4 for v in self.levels.values()):
            raise ValueError("levels must be in 0..4")
        ordered = [self.levels[s] for s in STATES]
        if ordered != sorted(ordered):
            raise ValueError("levels must not decrease with state severity")
        return self


def load_params(path: str | Path | None = None) -> JonasParams:
    """Load dial parameters from ``path`` or the packaged ``jonas.yaml``."""
    if path is None:
        text = resources.files("kairos.jonas").joinpath("jonas.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return JonasParams.model_validate(yaml.safe_load(text))


def classify(dgdt: float, dgdt_lo: float, test: SustainedResult) -> JonasState:
    """Map the latest ``dG/dt``, its lower bound and the sustained test to a state."""
    if not dgdt > 0:
        return "closing"
    if test.sustained:
        return "sustained_robust" if dgdt_lo > 0 else "sustained"
    if test.run_rule:
        return "persistent"
    return "widening"


def jonas_dial(
    frame: SeriesFrame,
    params: JonasParams | None = None,
    *,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> DialReading:
    """Read the Jonas dial from a ``SeriesFrame`` of C and H series.

    Args:
        frame: ``SeriesFrame`` with ``source_group`` ``"C"``/``"H"`` rows.
        params: Dial parameters; defaults to the packaged ``jonas.yaml``.
        seed: Bootstrap root seed (default ``Settings.seed``).
        n_jobs: Bootstrap workers.

    Returns:
        A :class:`~kairos.types.DialReading` with ``value = dG/dt`` (log units per
        year) at the latest month, its interval, and ``detail`` holding the state,
        run length, test statistics, the laggard H component and the closing rate.
    """
    p = params or load_params()
    vel = gap_velocity(
        frame,
        p.base_year,
        window=p.window,
        n_boot=p.n_boot,
        level=p.level,
        seed=seed,
        n_jobs=n_jobs,
    )
    valid = vel.dropna(subset=["dgdt"])
    if valid.empty:
        raise ValueError(f"need at least {p.window} months of overlapping C and H data")
    last = valid.iloc[-1]
    test = sustained_positive(valid, min_run=p.min_run, alpha=p.alpha, months=p.test_months)
    lo, hi = float(last["dgdt_lo"]), float(last["dgdt_hi"])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo = hi = float(last["dgdt"])
    state = classify(float(last["dgdt"]), lo, test)
    att = component_attribution(frame, p.base_year, months=p.scenario_months)
    sc = closing_scenarios(frame, p.base_year, months=p.scenario_months, level=p.level)
    return DialReading(
        dial=DIAL_NAME,
        value=float(last["dgdt"]),
        level=p.levels[state],
        ci_lo=min(lo, float(last["dgdt"])),
        ci_hi=max(hi, float(last["dgdt"])),
        as_of=last["ds"].date(),
        detail={
            "state": state,
            "run_length": test.run_length,
            "gap_relative": float(last["gap"]),
            "base_year": p.base_year,
            "interval_level": p.level,
            "sustained": test.as_dict(),
            "laggard": str(att.loc[0, "series_id"]),
            "required_h_rate": sc.required_rate,
            "required_h_pct": sc.required_pct,
            "shortfall": sc.shortfall,
            "params_version": p.version,
        },
    )
