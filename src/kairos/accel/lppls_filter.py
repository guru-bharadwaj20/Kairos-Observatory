"""LPPLS qualification filter (K-5.08).

A fit only counts as a bubble signature when its parameters are in the range found
in empirical studies (Sornette et al. 2015, Demos and Sornette 2017):

* ``m ∈ [0.1, 0.9]`` (super-exponential but finite price at ``t_c``)
* ``ω ∈ [6, 13]`` (log-periodic frequency)
* damping ``m|B| / (ω|C|) ≥ 0.8`` (the hazard rate stays positive)
* oscillations ``(ω/2π) ln((t_c − t1)/(t_c − t2)) ≥ 2.5`` (the oscillations are
  not just a trend artefact)
* ``B < 0`` for a positive (upward) bubble
* ``t_c`` close to the window end: ``(t_c − t2)/(t2 − t1) ≤ 0.2``
* optional relative fit error cap

The thresholds are the defaults of :class:`LPPLSFilter`; pass another instance to
change them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kairos.accel.lppls import LPPLSFit

__all__ = ["DEFAULT_FILTER", "LPPLSFilter", "Qualification", "qualify"]


@dataclass(frozen=True)
class LPPLSFilter:
    """Bounds a fit must satisfy to qualify.

    Attributes:
        m_range: Allowed ``m``.
        omega_range: Allowed ``ω``.
        min_damping: Minimum damping ratio.
        min_oscillations: Minimum oscillations inside the window.
        max_tc_ahead: Maximum ``(t_c − t2)/(t2 − t1)``.
        require_negative_b: Require ``B < 0`` (positive bubble).
        max_rmse: Optional cap on the log RMSE.
    """

    m_range: tuple[float, float] = (0.1, 0.9)
    omega_range: tuple[float, float] = (6.0, 13.0)
    min_damping: float = 0.8
    min_oscillations: float = 2.5
    max_tc_ahead: float = 0.2
    require_negative_b: bool = True
    max_rmse: float | None = None


DEFAULT_FILTER = LPPLSFilter()


@dataclass(frozen=True)
class Qualification:
    """Outcome of :func:`qualify`.

    Attributes:
        qualified: Every check passed.
        checks: Name → pass flag for each condition.
    """

    qualified: bool
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def failed(self) -> list[str]:
        """Names of the failed checks."""
        return [k for k, ok in self.checks.items() if not ok]


def qualify(fit: LPPLSFit, flt: LPPLSFilter = DEFAULT_FILTER) -> Qualification:
    """Apply the filter conditions to one fit."""
    dt = fit.t2 - fit.t1
    checks = {
        "m": flt.m_range[0] <= fit.m <= flt.m_range[1],
        "omega": flt.omega_range[0] <= fit.omega <= flt.omega_range[1],
        "damping": fit.damping >= flt.min_damping,
        "oscillations": fit.n_oscillations >= flt.min_oscillations,
        "tc": 0 < fit.tc - fit.t2 <= flt.max_tc_ahead * dt,
    }
    if flt.require_negative_b:
        checks["b"] = fit.b < 0
    if flt.max_rmse is not None:
        checks["rmse"] = fit.rmse <= flt.max_rmse
    return Qualification(qualified=all(checks.values()), checks=checks)
