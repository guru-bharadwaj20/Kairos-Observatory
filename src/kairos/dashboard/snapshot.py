"""The published snapshot: the only input the dashboard reads.

The weekly pipeline (K-12.01) writes one JSON document per run. The dashboard
never fits models or touches raw data; it renders whatever the snapshot says.
That keeps the public page reproducible from a single hashed artifact.

Schema version 1 (``schema_version: 1``)::

    {
      "schema_version": 1,
      "generated_at": "2026-11-06T00:00:00Z",   # ISO-8601 UTC
      "data_vintage": "2026-11-01",              # cutoff of the data used
      "is_demo": false,                          # true only for the fixture
      "parameters": {"lambda": 9.0, "tripwires_version": "1.0.0",
                     "vinge_tolerance_factor": 2.0},
      "alert": {"level": 0-4, "name": str?, "response": str?,
                "lead_time": {"median": 31.0, "lower": 18.0, "upper": 52.0,
                              "interval": 0.9, "unit": "months"} | null},
      "dials": [DialReading, ...],               # exactly the four dials
      "capability": SeriesFrame,                 # C(t)
      "human_capacity": SeriesFrame,             # H(t)
      "acceleration": [AccelerationFit, ...],    # one per C-side series
      "vinge_horizon": {"t": [...], "h_star": [...], "unit": "months",
                        "by_tolerance": {"1.5": [...], "3.0": [...]}},  # optional
      "ledger": [LedgerEntry, ...]
    }

    DialReading   = {"dial": "acceleration"|"recursion"|"vinge_horizon"|"jonas_gap",
                     "symbol": str, "value": float, "unit": str,
                     "state": "calm"|"danger", "p_danger": float in [0, 1],
                     "history": {"t": [date, ...], "value": [float, ...]}}
    SeriesFrame   = {"name": str, "unit": str, "t": [date, ...],
                     "value": [float, ...]}       # base-year normalised (D8)
    AccelerationFit = {"series": str, "unit": str, "t": [...], "observed": [...],
                       "fit": [...], "lower": [...], "upper": [...],
                       "changepoints": [date, ...], "delta_samples": [float, ...]}
    LedgerEntry   = {"id": str, "registered_at": iso, "statement": str,
                     "hash": hex sha256, "prev_hash": hex | null,
                     "proof": "verified"|"pending"|"failed",
                     "outcome_date": date, "score": float | null}

Field names for dials follow ``kairos.types.DialReading``; the loader accepts
extra keys so producers may add fields without breaking older dashboards.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

SCHEMA_VERSION = 1

DialName = Literal["acceleration", "recursion", "vinge_horizon", "jonas_gap"]
DIAL_ORDER: tuple[DialName, ...] = ("acceleration", "recursion", "vinge_horizon", "jonas_gap")
#: Names used by the dial modules (``DialReading.dial``) mapped to dashboard names.
DIAL_ALIASES: dict[str, DialName] = {"vinge": "vinge_horizon", "jonas": "jonas_gap"}
DialState = Literal["calm", "danger"]
ProofStatus = Literal["verified", "pending", "failed"]


class SnapshotError(ValueError):
    """The snapshot does not match the published schema."""


@dataclass(frozen=True)
class SeriesFrame:
    """A dated series of floats (columns ``t`` and ``value``)."""

    name: str
    unit: str
    t: tuple[str, ...]
    value: tuple[float, ...]


@dataclass(frozen=True)
class DialReading:
    """One dial's current reading, state and recent history."""

    dial: DialName
    symbol: str
    value: float
    unit: str
    state: DialState
    p_danger: float
    history: SeriesFrame


@dataclass(frozen=True)
class LeadTime:
    """Collingridge lead time: median with its interval."""

    median: float
    lower: float
    upper: float
    interval: float
    unit: str


@dataclass(frozen=True)
class Alert:
    """The tripwire level and lead time."""

    level: int
    lead_time: LeadTime | None
    name: str | None = None  # overrides the default level name
    response: str | None = None  # overrides the default recommended response


@dataclass(frozen=True)
class AccelerationFit:
    """Prophet null fit, changepoints and δ bootstrap for one series."""

    series: str
    unit: str
    t: tuple[str, ...]
    observed: tuple[float, ...]
    fit: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    changepoints: tuple[str, ...]
    delta_samples: tuple[float, ...]


@dataclass(frozen=True)
class VingeHorizon:
    """h*(t), at the published tolerance and optionally others."""

    t: tuple[str, ...]
    h_star: tuple[float, ...]  # at the published tolerance
    unit: str
    #: h* recomputed at other tolerance factors (e.g. 1.5x, 2x, 3x), if published
    by_tolerance: Mapping[float, tuple[float, ...]] = field(default_factory=dict)

    def at(self, factor: float, published: float) -> tuple[float, ...]:
        """Return h* at a tolerance factor."""
        if factor in self.by_tolerance:
            return self.by_tolerance[factor]
        if factor == published:
            return self.h_star
        raise KeyError(factor)

    def tolerances(self, published: float) -> list[float]:
        """Return every tolerance factor with published h*."""
        return sorted({published, *self.by_tolerance})


@dataclass(frozen=True)
class LedgerEntry:
    """One pre-registered forecast as published in the ledger."""

    id: str
    registered_at: str
    statement: str
    hash: str
    prev_hash: str | None
    proof: ProofStatus
    outcome_date: str
    score: float | None


@dataclass(frozen=True)
class Parameters:
    """Published parameters the snapshot was computed with."""

    lam: float
    tripwires_version: str
    vinge_tolerance_factor: float


@dataclass(frozen=True)
class Snapshot:
    """The whole published snapshot."""

    schema_version: int
    generated_at: str
    data_vintage: str
    is_demo: bool
    parameters: Parameters
    alert: Alert
    dials: tuple[DialReading, ...]
    capability: SeriesFrame
    human_capacity: SeriesFrame
    acceleration: tuple[AccelerationFit, ...]
    vinge_horizon: VingeHorizon
    ledger: tuple[LedgerEntry, ...]

    def dial(self, name: DialName) -> DialReading:
        """Return the reading for one dial."""
        for d in self.dials:
            if d.dial == name:
                return d
        raise KeyError(name)


# --- parsing helpers -------------------------------------------------------


def _obj(raw: object, where: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise SnapshotError(f"{where}: expected an object")
    return cast(Mapping[str, object], raw)


def _get(m: Mapping[str, object], key: str, where: str) -> object:
    if key not in m:
        raise SnapshotError(f"{where}: missing field '{key}'")
    return m[key]


def _str(m: Mapping[str, object], key: str, where: str) -> str:
    v = _get(m, key, where)
    if not isinstance(v, str):
        raise SnapshotError(f"{where}.{key}: expected a string")
    return v


def _num(m: Mapping[str, object], key: str, where: str) -> float:
    v = _get(m, key, where)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise SnapshotError(f"{where}.{key}: expected a number")
    return float(v)


def _opt_num(m: Mapping[str, object], key: str, where: str) -> float | None:
    return None if m.get(key) is None else _num(m, key, where)


def _list(m: Mapping[str, object], key: str, where: str) -> Sequence[object]:
    v = _get(m, key, where)
    if not isinstance(v, list):
        raise SnapshotError(f"{where}.{key}: expected a list")
    return cast(list[object], v)


def _strs(m: Mapping[str, object], key: str, where: str) -> tuple[str, ...]:
    out = _list(m, key, where)
    if not all(isinstance(x, str) for x in out):
        raise SnapshotError(f"{where}.{key}: expected a list of strings")
    return tuple(cast(Sequence[str], out))


def _nums(m: Mapping[str, object], key: str, where: str) -> tuple[float, ...]:
    out = _list(m, key, where)
    if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in out):
        raise SnapshotError(f"{where}.{key}: expected a list of numbers")
    return tuple(float(cast(float, x)) for x in out)


def _same_len(where: str, *cols: Sequence[object]) -> None:
    if len({len(c) for c in cols}) > 1:
        raise SnapshotError(f"{where}: columns have different lengths")


def _series(raw: object, where: str, name: str | None = None) -> SeriesFrame:
    m = _obj(raw, where)
    t, v = _strs(m, "t", where), _nums(m, "value", where)
    _same_len(where, t, v)
    return SeriesFrame(
        name=name if name is not None else _str(m, "name", where),
        unit=str(m.get("unit", "")),
        t=t,
        value=v,
    )


def _dial(raw: object, where: str) -> DialReading:
    m = _obj(raw, where)
    name = _str(m, "dial", where)
    name = DIAL_ALIASES.get(name, name)
    if name not in DIAL_ORDER:
        raise SnapshotError(f"{where}.dial: unknown dial '{name}'")
    state = _str(m, "state", where)
    if state not in ("calm", "danger"):
        raise SnapshotError(f"{where}.state: must be 'calm' or 'danger'")
    p = _num(m, "p_danger", where)
    if not 0.0 <= p <= 1.0:
        raise SnapshotError(f"{where}.p_danger: must lie in [0, 1]")
    return DialReading(
        dial=name,
        symbol=_str(m, "symbol", where),
        value=_num(m, "value", where),
        unit=str(m.get("unit", "")),
        state=cast(DialState, state),
        p_danger=p,
        history=_series(_get(m, "history", where), f"{where}.history", name=name),
    )


def _alert(raw: object) -> Alert:
    m = _obj(raw, "alert")
    level = _num(m, "level", "alert")
    if level not in (0, 1, 2, 3, 4):
        raise SnapshotError("alert.level: must be an integer 0-4")
    lt_raw = m.get("lead_time")
    lead: LeadTime | None = None
    if lt_raw is not None:
        lt = _obj(lt_raw, "alert.lead_time")
        w = "alert.lead_time"
        lead = LeadTime(
            median=_num(lt, "median", w),
            lower=_num(lt, "lower", w),
            upper=_num(lt, "upper", w),
            interval=_num(lt, "interval", w),
            unit=_str(lt, "unit", w),
        )
        if not lead.lower <= lead.median <= lead.upper:
            raise SnapshotError(f"{w}: need lower <= median <= upper")
    name, response = m.get("name"), m.get("response")
    return Alert(
        level=int(level),
        lead_time=lead,
        name=name if isinstance(name, str) else None,
        response=response if isinstance(response, str) else None,
    )


def _accel(raw: object, where: str) -> AccelerationFit:
    m = _obj(raw, where)
    a = AccelerationFit(
        series=_str(m, "series", where),
        unit=str(m.get("unit", "")),
        t=_strs(m, "t", where),
        observed=_nums(m, "observed", where),
        fit=_nums(m, "fit", where),
        lower=_nums(m, "lower", where),
        upper=_nums(m, "upper", where),
        changepoints=_strs(m, "changepoints", where),
        delta_samples=_nums(m, "delta_samples", where),
    )
    _same_len(where, a.t, a.observed, a.fit, a.lower, a.upper)
    return a


def _ledger(raw: object, where: str) -> LedgerEntry:
    m = _obj(raw, where)
    proof = _str(m, "proof", where)
    if proof not in ("verified", "pending", "failed"):
        raise SnapshotError(f"{where}.proof: must be verified, pending or failed")
    prev = m.get("prev_hash")
    if prev is not None and not isinstance(prev, str):
        raise SnapshotError(f"{where}.prev_hash: expected a string or null")
    return LedgerEntry(
        id=_str(m, "id", where),
        registered_at=_str(m, "registered_at", where),
        statement=_str(m, "statement", where),
        hash=_str(m, "hash", where),
        prev_hash=prev,
        proof=cast(ProofStatus, proof),
        outcome_date=_str(m, "outcome_date", where),
        score=_opt_num(m, "score", where),
    )


def parse_snapshot(raw: object) -> Snapshot:
    """Validate a decoded JSON document and return a typed snapshot."""
    m = _obj(raw, "snapshot")
    version = _num(m, "schema_version", "snapshot")
    if version != SCHEMA_VERSION:
        raise SnapshotError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")
    p = _obj(_get(m, "parameters", "snapshot"), "parameters")
    dials = tuple(_dial(d, f"dials[{i}]") for i, d in enumerate(_list(m, "dials", "snapshot")))
    if sorted(d.dial for d in dials) != sorted(DIAL_ORDER):
        raise SnapshotError("dials: need exactly one reading per dial")
    dials = tuple(sorted(dials, key=lambda d: DIAL_ORDER.index(d.dial)))
    vh = _obj(_get(m, "vinge_horizon", "snapshot"), "vinge_horizon")
    by_tol: dict[float, tuple[float, ...]] = {}
    raw_tol = _obj(vh.get("by_tolerance", {}), "vinge_horizon.by_tolerance")
    for key in raw_tol:
        try:
            factor = float(key)
        except ValueError as exc:
            raise SnapshotError(f"vinge_horizon.by_tolerance: key '{key}' is not a number") from exc
        by_tol[factor] = _nums(raw_tol, key, "vinge_horizon.by_tolerance")
    vinge = VingeHorizon(
        t=_strs(vh, "t", "vinge_horizon"),
        h_star=_nums(vh, "h_star", "vinge_horizon"),
        unit=str(vh.get("unit", "months")),
        by_tolerance=by_tol,
    )
    _same_len("vinge_horizon", vinge.t, vinge.h_star, *by_tol.values())
    return Snapshot(
        schema_version=SCHEMA_VERSION,
        generated_at=_str(m, "generated_at", "snapshot"),
        data_vintage=_str(m, "data_vintage", "snapshot"),
        is_demo=bool(m.get("is_demo", False)),
        parameters=Parameters(
            lam=_num(p, "lambda", "parameters"),
            tripwires_version=_str(p, "tripwires_version", "parameters"),
            vinge_tolerance_factor=_num(p, "vinge_tolerance_factor", "parameters"),
        ),
        alert=_alert(_get(m, "alert", "snapshot")),
        dials=dials,
        capability=_series(_get(m, "capability", "snapshot"), "capability"),
        human_capacity=_series(_get(m, "human_capacity", "snapshot"), "human_capacity"),
        acceleration=tuple(
            _accel(a, f"acceleration[{i}]")
            for i, a in enumerate(_list(m, "acceleration", "snapshot"))
        ),
        vinge_horizon=vinge,
        ledger=tuple(
            _ledger(e, f"ledger[{i}]") for i, e in enumerate(_list(m, "ledger", "snapshot"))
        ),
    )


def load_snapshot(path: str | Path) -> Snapshot:
    """Read and validate a published snapshot file."""
    try:
        raw: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"{path}: not valid JSON ({exc})") from exc
    return parse_snapshot(raw)


DEMO_SNAPSHOT_PATH = Path(__file__).with_name("fixtures") / "demo_snapshot.json"


def load_demo() -> Snapshot:
    """The synthetic fixture shipped with the package (``is_demo: true``)."""
    return load_snapshot(DEMO_SNAPSHOT_PATH)
