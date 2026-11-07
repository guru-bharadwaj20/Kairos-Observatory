"""Score matured forecasts and append the scores to the ledger (K-10.05).

A forecast *matures* when its last target date has passed and every target date
has an observed value. It is then scored once, and the score is appended as a
``score`` entry that names the forecast entry's hash; nothing is ever edited.

Forecasts are conditional on "no new intervention" (K-10.03). If interventions
were logged after a forecast was registered and took effect inside its horizon,
the score is flagged ``counterfactual``: the forecast then describes the world
*without* those interventions, and the gap ``log(y / median)`` estimates their
effect (Insight 4).

The proper scores here work directly on the stored quantiles:

* CRPS is approximated by the average quantile (pinball) loss,
  ``CRPS ≈ (2 / K) · Σ_k ρ_τk(y − q_k)``, exact as ``K → ∞`` for evenly spaced
  levels (Gneiting and Raftery 2007; Laio and Tamea 2007);
* PIT is the forecast CDF at ``y``, linearly interpolated between quantiles;
* coverage counts whether ``y`` falls inside the central interval.

They have the same definitions as the backtest scoring module (K-9.08) and can be
swapped for it once that lands.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from kairos.ledger.chain import Ledger, LedgerEntry
from kairos.ledger.conditional import ConditionalClaim, EventAssumption
from kairos.ledger.forecast import Forecast, ForecastPoint

__all__ = [
    "ClaimOutcome",
    "ForecastScore",
    "PointScore",
    "crps_quantiles",
    "interventions_since",
    "log_intervention",
    "pit_quantiles",
    "score_forecast",
    "score_matured",
]


# --------------------------------------------------------------------- scores
def crps_quantiles(levels: Sequence[float], values: Sequence[float], y: float) -> float:
    """Approximate CRPS of a quantile forecast at observation ``y``."""
    if len(levels) != len(values) or len(levels) == 0:
        raise ValueError("levels and values must be non-empty and the same length")
    total = 0.0
    for tau, q in zip(levels, values, strict=True):
        diff = y - q
        total += max(tau * diff, (tau - 1.0) * diff)
    return 2.0 * total / len(levels)


def pit_quantiles(levels: Sequence[float], values: Sequence[float], y: float) -> float:
    """Return the forecast CDF at ``y`` by linear interpolation between quantiles.

    Below the lowest quantile the value is that quantile's level, above the
    highest it is the highest level (the tails are not modelled).
    """
    xs = np.asarray(values, dtype=float)
    ps = np.asarray(levels, dtype=float)
    if np.any(np.diff(xs) == 0):  # flat steps: keep interpolation well defined
        xs = xs + np.arange(len(xs)) * 1e-12 * max(1.0, float(np.abs(xs).max()))
    return float(np.interp(y, xs, ps))


# --------------------------------------------------------------------- models
class PointScore(BaseModel):
    """Score of one forecast point.

    Attributes:
        ds: Target date.
        y: Observed value.
        median: Forecast median (``0.5`` quantile), if present.
        crps: Approximate CRPS (raw units).
        pit: Probability integral transform value.
        covered: Whether ``y`` lies in the central interval (``None`` if the
            forecast lacks its quantiles).
        log_gap: ``log(y / median)`` for positive values, else ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ds: date
    y: float
    median: float | None = None
    crps: float
    pit: float
    covered: bool | None = None
    log_gap: float | None = None


class ClaimOutcome(BaseModel):
    """Resolution of a conditional claim.

    Attributes:
        threshold: The claim's ``Y``.
        crossed: First observed date with ``y >= threshold`` (``None`` if never).
        within_interval: Whether ``crossed`` lies in ``[earliest, latest]``, where an
            open ``latest`` means "after the horizon" (so not crossing counts as inside).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float
    crossed: date | None = None
    within_interval: bool


class ForecastScore(BaseModel):
    """Score of one matured forecast; the payload of a ``score`` entry.

    Attributes:
        forecast_entry: ``entry_hash`` of the scored forecast entry.
        forecast_hash: Content hash of the forecast.
        series_id: Series forecast.
        model: Model that produced it.
        as_of: Data vintage of the forecast.
        scored_on: Date the score was computed.
        coverage_level: Nominal coverage of the central interval used.
        points: Per-point scores.
        mean_crps: Mean CRPS over points.
        coverage: Fraction of points inside the central interval.
        counterfactual: Interventions took effect after registration.
        interventions: ``entry_hash`` of each such intervention entry.
        mean_log_gap: Mean ``log(y / median)``; under ``counterfactual`` it
            estimates the interventions' effect.
        claims: Resolution of each conditional claim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_entry: str
    forecast_hash: str
    series_id: str
    model: str
    as_of: date
    scored_on: date
    coverage_level: float = Field(gt=0, lt=1)
    points: tuple[PointScore, ...]
    mean_crps: float
    coverage: float | None = None
    counterfactual: bool = False
    interventions: tuple[str, ...] = ()
    mean_log_gap: float | None = None
    claims: tuple[ClaimOutcome, ...] = ()


# -------------------------------------------------------------- interventions
def log_intervention(
    ledger: Ledger,
    event: EventAssumption,
    *,
    series_ids: Iterable[str] = (),
    recorded_at: datetime | None = None,
) -> LedgerEntry:
    """Append an ``intervention`` entry.

    Args:
        ledger: Ledger to append to.
        event: The intervention (name, effective date, description).
        series_ids: Series it is expected to affect; empty means all series.
        recorded_at: Override of the append time (tests, backfills).
    """
    payload = {"event": event.model_dump(mode="json"), "series_ids": sorted(series_ids)}
    return ledger.append("intervention", payload, recorded_at=recorded_at)


def interventions_since(
    entries: Sequence[LedgerEntry], forecast_entry: LedgerEntry
) -> list[LedgerEntry]:
    """Return interventions logged after ``forecast_entry`` that affect it.

    An intervention affects a forecast when it was appended later in the ledger,
    takes effect no later than the forecast's horizon end, targets its series (or
    all series) and is not one of the events the forecast already assumed.
    """
    forecast = forecast_entry.forecast()
    assumed = {e.name for e in forecast.events or ()}
    hits: list[LedgerEntry] = []
    for entry in entries:
        if entry.kind != "intervention" or entry.seq <= forecast_entry.seq:
            continue
        event = EventAssumption.model_validate(entry.payload["event"])
        series = entry.payload.get("series_ids") or []
        if event.name in assumed or event.ds > forecast.horizon_end:
            continue
        if series and forecast.series_id not in series:
            continue
        hits.append(entry)
    return hits


# -------------------------------------------------------------------- scoring
def _actuals_map(actuals: pd.DataFrame, series_id: str) -> dict[date, float]:
    rows = actuals.loc[actuals["series_id"] == series_id, ["ds", "y"]].dropna()
    return {pd.Timestamp(ds).date(): float(y) for ds, y in zip(rows["ds"], rows["y"], strict=True)}


def _score_point(point: ForecastPoint, y: float, coverage_level: float) -> PointScore:
    levels = point.levels()
    values = [point.at(lv) for lv in levels]
    lo_level = round((1 - coverage_level) / 2, 6)
    hi_level = round((1 + coverage_level) / 2, 6)
    covered: bool | None = None
    if lo_level in levels and hi_level in levels:
        covered = point.at(lo_level) <= y <= point.at(hi_level)
    median = point.at(0.5) if 0.5 in levels else None
    log_gap = math.log(y / median) if median is not None and median > 0 and y > 0 else None
    return PointScore(
        ds=point.ds,
        y=y,
        median=median,
        crps=crps_quantiles(levels, values, y),
        pit=pit_quantiles(levels, values, y),
        covered=covered,
        log_gap=log_gap,
    )


def _resolve_claim(claim: ConditionalClaim, observed: Mapping[date, float]) -> ClaimOutcome:
    crossed = next(
        (d for d in sorted(observed) if d <= claim.horizon_end and observed[d] >= claim.threshold),
        None,
    )
    if crossed is None:
        inside = claim.latest is None
    else:
        lo_ok = claim.earliest is not None and claim.earliest <= crossed
        hi_ok = claim.latest is None or crossed <= claim.latest
        inside = lo_ok and hi_ok
    return ClaimOutcome(threshold=claim.threshold, crossed=crossed, within_interval=inside)


def score_forecast(
    forecast_entry: LedgerEntry,
    actuals: pd.DataFrame,
    *,
    today: date,
    entries: Sequence[LedgerEntry] = (),
    coverage_level: float = 0.9,
) -> ForecastScore | None:
    """Score one forecast entry, or return ``None`` if it has not matured.

    Args:
        forecast_entry: A ``forecast`` ledger entry.
        actuals: Observed values as a ``SeriesFrame`` (``ds``, ``y``, ``series_id``).
        today: Scoring date; the horizon end must not be after it.
        entries: The whole ledger, used to find later interventions.
        coverage_level: Central interval to check coverage for.
    """
    forecast: Forecast = forecast_entry.forecast()
    if forecast.horizon_end > today:
        return None
    observed = _actuals_map(actuals, forecast.series_id)
    if any(p.ds not in observed for p in forecast.points):
        return None
    scores = tuple(_score_point(p, observed[p.ds], coverage_level) for p in forecast.points)
    covered = [s.covered for s in scores if s.covered is not None]
    gaps = [s.log_gap for s in scores if s.log_gap is not None]
    later = interventions_since(entries, forecast_entry)
    return ForecastScore(
        forecast_entry=forecast_entry.entry_hash,
        forecast_hash=forecast_entry.payload["forecast_hash"],
        series_id=forecast.series_id,
        model=forecast.model,
        as_of=forecast.as_of,
        scored_on=today,
        coverage_level=coverage_level,
        points=scores,
        mean_crps=float(np.mean([s.crps for s in scores])),
        coverage=float(np.mean(covered)) if covered else None,
        counterfactual=bool(later),
        interventions=tuple(e.entry_hash for e in later),
        mean_log_gap=float(np.mean(gaps)) if gaps else None,
        claims=tuple(_resolve_claim(c, observed) for c in forecast.claims or ()),
    )


def _superseded(entries: Sequence[LedgerEntry]) -> set[str]:
    return {e.payload["supersedes"] for e in entries if e.kind == "correction"}


def score_matured(
    ledger: Ledger,
    actuals: pd.DataFrame,
    *,
    today: date,
    coverage_level: float = 0.9,
    append: bool = True,
    recorded_at: datetime | None = None,
) -> list[ForecastScore]:
    """Score every matured, not-yet-scored, not-superseded forecast in ``ledger``.

    With ``append`` (the default) each score is appended as a ``score`` entry, so
    running this twice scores nothing new the second time.
    """
    entries = ledger.entries()
    done = {e.payload["forecast_entry"] for e in entries if e.kind == "score"}
    skip = done | _superseded(entries)
    results: list[ForecastScore] = []
    for entry in entries:
        if entry.kind != "forecast" or entry.entry_hash in skip:
            continue
        score = score_forecast(
            entry, actuals, today=today, entries=entries, coverage_level=coverage_level
        )
        if score is None:
            continue
        results.append(score)
        if append:
            payload: dict[str, Any] = score.model_dump(mode="json", exclude_none=True)
            ledger.append("score", payload, recorded_at=recorded_at)
    return results
