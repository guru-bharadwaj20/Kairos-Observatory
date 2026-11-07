"""Stage functions behind ``kairos run weekly`` and ``kairos run validation``.

Each function has the stage signature ``(ctx: RunContext) -> Mapping | None``
expected by :mod:`kairos.commands.run`. They are thin glue over the phase
packages; every scientific choice stays in those packages.

Layout under ``ctx.root``::

    data/                    refreshed dataset and MANIFEST.json   (refresh)
    ledger/forecasts.jsonl   hash-chained forecast ledger           (ledger)
    ledger/proofs/           OpenTimestamps proofs, when reachable  (ledger)
    build/snapshot.json      schema-v1 dashboard snapshot           (export)
    build/site/              static dashboard                       (export)

A dial that cannot be read for lack of data (a ``ValueError`` from its package)
is recorded as skipped with the reason, and the run carries on. Any other
exception is a bug and fails the stage.

Set ``KAIROS_OFFLINE=1`` to forbid network access (live data and timestamp
calendars). ``--cpu-only`` only selects a reduced ``κ`` ensemble; it does not
change data access.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from kairos.logging import get_logger
from kairos.types import DialReading

if TYPE_CHECKING:
    from kairos.commands.run import RunContext

__all__ = [
    "CPU_KAPPA",
    "dials",
    "export",
    "fit",
    "governor",
    "ledger",
    "refresh",
]

log = get_logger(__name__)

DIAL_NAMES = ("acceleration", "recursion", "vinge", "jonas")
LEDGER_SERIES = "idx.C"
LEDGER_HORIZON = 12
FIT_MIN_OBS = 24
CPU_KAPPA = {"n_seeds": 3, "n_permutations": 3, "epochs": 30}
"""Reduced κ ensemble for ``--cpu-only`` runs (about a minute on a laptop)."""
ACCEL_N_BOOT = {True: 200, False: 1000}
VINGE_HORIZON = 24
VINGE_SERIES = ("idx.C", "c.frontier_compute")
DRIVER_PREFIX = "exo."

_SYMBOLS = {"acceleration": "δ", "recursion": "κ", "vinge": "h*", "jonas": "dG/dt"}
_UNITS = {"acceleration": "", "recursion": "", "vinge": "months", "jonas": "per year"}


def _offline(ctx: RunContext) -> bool:
    return os.environ.get("KAIROS_OFFLINE", "") not in ("", "0")


def _data_root(ctx: RunContext) -> Path:
    return ctx.root / "data"


def _frame(datasets: Mapping[str, Any], ids: list[str]) -> pd.DataFrame:
    return pd.concat([datasets[i].frame for i in ids], ignore_index=True)


def _load(ctx: RunContext) -> dict[str, Any]:
    from kairos.data.store import read_all

    datasets = read_all(_data_root(ctx))
    if not datasets:
        raise FileNotFoundError(f"no processed series under {_data_root(ctx)}; run refresh")
    return datasets


# --------------------------------------------------------------------------- refresh


def refresh(ctx: RunContext) -> Mapping[str, Any]:
    """Rebuild ``data/`` from live sources, or from committed snapshots offline."""
    import httpx

    from kairos.data.pipeline import refresh as data_refresh

    offline = _offline(ctx)
    note = "offline (committed snapshots)" if offline else "live"
    try:
        res = data_refresh(_data_root(ctx), offline=offline)
    except (httpx.HTTPError, OSError) as exc:
        if offline:
            raise
        log.warning("live refresh failed (%s); falling back to snapshots", exc)
        res = data_refresh(_data_root(ctx), offline=True)
        note = f"offline fallback after {type(exc).__name__}"
    return {
        "detail": f"{len(res.series)} series, {res.n_events} events, {note}",
        "series": res.series,
        "digest": res.digest,
        "offline": offline,
    }


# --------------------------------------------------------------------------- fit


def fit(ctx: RunContext) -> Mapping[str, Any]:
    """Fit the Prophet null model to every C/H series with enough history.

    NeuralProphet is fitted inside the recursion dial (``κ``), not here.
    """
    from kairos.models.fit_all import fit_all

    datasets = _load(ctx)
    ids = [
        sid
        for sid, d in datasets.items()
        if d.meta.side in ("C", "H")
        and int(d.frame["y"].notna().sum()) >= FIT_MIN_OBS
        and bool((d.frame["y"].dropna() > 0).all())
    ]
    frame = _frame(datasets, ids)[["ds", "y", "series_id"]]
    res = fit_all(frame, horizon=LEDGER_HORIZON, n_jobs=1 if len(ids) <= 4 else None)
    return {
        "detail": f"Prophet null model on {len(res.fits)} series ({', '.join(res.fits)})",
        "fits": res.fits,
    }


# --------------------------------------------------------------------------- dials


def _acceleration(ctx: RunContext, datasets: Mapping[str, Any]) -> DialReading:
    from kairos.accel.bootstrap import BootstrapResult, block_bootstrap_frame
    from kairos.accel.dial import acceleration_dial

    results: dict[str, BootstrapResult] = {}
    for sid, d in datasets.items():
        if d.meta.side != "C":
            continue
        try:
            results[sid] = block_bootstrap_frame(
                d.frame, sid, n_boot=ACCEL_N_BOOT[ctx.cpu_only], n_jobs=1
            )
        except ValueError as exc:
            log.info("acceleration: %s skipped (%s)", sid, exc)
    groups = {sid: str(datasets[sid].meta.source_group) for sid in results}
    as_of = max(pd.Timestamp(datasets[s].frame["ds"].max()) for s in results).date()
    reading = acceleration_dial(results, groups, as_of=as_of)
    detail = dict(reading.detail)
    detail["n_independent_groups"] = detail["n_groups"]
    if LEDGER_SERIES in results:
        detail["delta_samples_" + LEDGER_SERIES] = [
            float(x) for x in results[LEDGER_SERIES].delta if math.isfinite(float(x))
        ]
    return reading.model_copy(update={"detail": detail})


def _recursion(ctx: RunContext, datasets: Mapping[str, Any]) -> DialReading:
    from kairos.neural.kappa import estimate_kappa, kappa_reading
    from kairos.neural.paired import PairedSpec
    from kairos.neural.regressors import DEFAULT_DRIVERS, design_frame

    ids = [LEDGER_SERIES] + [DRIVER_PREFIX + d.series_id for d in DEFAULT_DRIVERS]
    missing = [i for i in ids if i not in datasets]
    if missing:
        raise ValueError(f"missing series {missing}")
    frame = _frame(datasets, ids)
    frame["series_id"] = frame["series_id"].str.removeprefix(DRIVER_PREFIX)
    design = design_frame(frame, LEDGER_SERIES)
    if ctx.cpu_only:
        spec = PairedSpec(
            config=PairedSpec().config.model_copy(update={"epochs": CPU_KAPPA["epochs"]})
        )
        result = estimate_kappa(
            design,
            spec,
            n_seeds=CPU_KAPPA["n_seeds"],
            n_permutations=CPU_KAPPA["n_permutations"],
            device="cpu",
        )
    else:
        result = estimate_kappa(design, device=ctx.device)
    reading = kappa_reading(result, pd.Timestamp(design["ds"].max()).date())
    if ctx.cpu_only:
        detail = dict(reading.detail) | {"reduced_ensemble": dict(CPU_KAPPA)}
        reading = reading.model_copy(update={"detail": detail})
    return reading


def _vinge(ctx: RunContext, datasets: Mapping[str, Any]) -> DialReading:
    from kairos.vinge.backtest import rolling_origin_backtest
    from kairos.vinge.baselines import LogDriftForecaster
    from kairos.vinge.dial import load_thresholds, spec_from_thresholds, vinge_dial
    from kairos.vinge.series import hstar_series

    ids = [s for s in VINGE_SERIES if s in datasets]
    if not ids:
        raise ValueError(f"none of {VINGE_SERIES} in the dataset")
    # One vintage only: replay it pseudo-real-time (drop the vintage column, which
    # would otherwise hide every row from every historical cutoff).
    frame = _frame(datasets, ids)[["ds", "y", "series_id"]]
    th = load_thresholds()
    bt = rolling_origin_backtest(frame, LogDriftForecaster(), VINGE_HORIZON, use_cache=False)
    hstar = hstar_series(bt, spec_from_thresholds(th))
    reading = vinge_dial(hstar, th)
    history = hstar.groupby("as_of")["h_star"].median()
    detail = dict(reading.detail) | {
        "history": {
            "t": [pd.Timestamp(t).date().isoformat() for t in history.index],
            "value": [float(v) for v in history.to_numpy()],
        },
        "tolerance_factor": th.tolerance_factor,
        "forecaster": "log-drift",
    }
    return reading.model_copy(update={"detail": detail})


def _jonas(ctx: RunContext, datasets: Mapping[str, Any]) -> DialReading:
    from kairos.jonas.dial import jonas_dial

    for sid in ("idx.C", "idx.H"):
        if sid not in datasets:
            raise ValueError(f"missing composite {sid}")
    frame = _frame(datasets, ["idx.C", "idx.H"])[["ds", "y", "series_id", "side"]]
    frame["source_group"] = frame.pop("side").astype(str)
    return jonas_dial(frame)


DIAL_FUNCS: dict[str, Callable[[RunContext, Mapping[str, Any]], DialReading]] = {
    "acceleration": _acceleration,
    "recursion": _recursion,
    "vinge": _vinge,
    "jonas": _jonas,
}


def dials(ctx: RunContext) -> Mapping[str, Any]:
    """Read the four dials; a dial short of data is skipped with its reason."""
    datasets = _load(ctx)
    readings: dict[str, DialReading] = {}
    skipped: dict[str, str] = {}
    for name, func in DIAL_FUNCS.items():
        try:
            readings[name] = func(ctx, datasets)
        except ValueError as exc:
            log.warning("dial %s skipped: %s", name, exc)
            skipped[name] = str(exc)
    parts = [f"{n}={r.value:.3g} (level {r.level})" for n, r in readings.items()]
    parts += [f"{n} skipped: {why}" for n, why in skipped.items()]
    return {"detail": "; ".join(parts), "readings": readings, "skipped": skipped}


# --------------------------------------------------------------------------- governor


def _readings(ctx: RunContext) -> dict[str, DialReading]:
    out: dict[str, DialReading] = dict(ctx.artifacts.get("dials", {}).get("readings", {}))
    return out


def governor(ctx: RunContext) -> Mapping[str, Any]:
    """Feed this week's readings to the ethical governor and set the tripwire level."""
    from kairos.tripwires.aggregator import Governor

    readings = _readings(ctx)
    alert = Governor(allow_normal_fallback=True).evaluate(readings.values())
    missing = f"; held: {', '.join(alert.missing)}" if alert.missing else ""
    return {
        "detail": f"level {alert.level} ({alert.name}), {alert.n_danger} dial(s) in danger"
        + missing,
        "alert": alert,
    }


# --------------------------------------------------------------------------- ledger


def _ledger_forecast(ctx: RunContext) -> Any:
    from kairos import __version__
    from kairos.ledger.forecast import Forecast, ForecastPoint

    fits = ctx.artifacts.get("fit", {}).get("fits", {})
    if LEDGER_SERIES not in fits:
        raise ValueError(f"no fit for {LEDGER_SERIES}; run the fit stage first")
    f = fits[LEDGER_SERIES]
    as_of = pd.Timestamp(f.history["ds"].max())
    ahead = f.forecast.loc[f.forecast["ds"] > as_of].head(LEDGER_HORIZON)
    points = tuple(
        ForecastPoint(
            ds=pd.Timestamp(r.ds).date(),
            quantiles={
                0.05: float(min(r.yhat_lower, r.yhat)),
                0.5: float(r.yhat),
                0.95: float(max(r.yhat_upper, r.yhat)),
            },
        )
        for r in ahead.itertuples()
    )
    dial_summary = {
        n: {"value": r.value, "level": r.level, "ci": [r.ci_lo, r.ci_hi]}
        for n, r in _readings(ctx).items()
        if math.isfinite(r.value) and math.isfinite(r.ci_lo) and math.isfinite(r.ci_hi)
    }
    alert = ctx.artifacts.get("governor", {}).get("alert")
    return Forecast(
        series_id=LEDGER_SERIES,
        model="prophet-null",
        as_of=as_of.date(),
        issued_at=datetime.now(UTC),
        unit=str(f.params.get("unit", "")) if isinstance(f.params, dict) else "",
        points=points,
        code_version=__version__,
        params={
            "run_id": ctx.run_id,
            "interval_width": 0.9,
            "dials": dial_summary,
            "tripwire_level": None if alert is None else alert.level,
        },
    )


def ledger(ctx: RunContext) -> Mapping[str, Any]:
    """Append this week's forecast (and its alert) to the hash-chained ledger."""
    from kairos.ledger.chain import Ledger, manifest_hash
    from kairos.ledger.timestamp import default_timestamper, stamp_entries

    book = Ledger(ctx.root / "ledger" / "forecasts.jsonl")
    forecast = _ledger_forecast(ctx)
    m_hash = manifest_hash(_data_root(ctx) / "MANIFEST.json")
    for entry in book.forecasts():
        p = entry.payload.get("forecast", {})
        same = (p.get("series_id"), p.get("model"), p.get("as_of")) == (
            forecast.series_id,
            forecast.model,
            forecast.as_of.isoformat(),
        )
        if same and entry.manifest_hash == m_hash:
            return {
                "detail": f"already registered as entry {entry.seq} ({entry.entry_hash[:12]})",
                "entry": entry,
                "ledger": book,
            }
    entry = book.append_forecast(forecast, manifest_hash=m_hash)
    proofs_dir = book.path.parent / "proofs"
    timestamper = None if os.environ.get("KAIROS_OFFLINE") else default_timestamper()
    proofs = stamp_entries([entry], timestamper, proofs_dir)
    return {
        "detail": f"entry {entry.seq} {entry.entry_hash[:12]}, proof {proofs[entry.seq]}",
        "entry": entry,
        "ledger": book,
    }


# --------------------------------------------------------------------------- export


def _finite(x: float, default: float = 0.0) -> float:
    return float(x) if math.isfinite(x) else default


def _series_json(datasets: Mapping[str, Any], sid: str, name: str) -> dict[str, Any]:
    if sid not in datasets:
        return {"name": name, "unit": "", "t": [], "value": []}
    f = datasets[sid].frame.dropna(subset=["y"])
    return {
        "name": name,
        "unit": str(datasets[sid].meta.unit),
        "t": [pd.Timestamp(t).date().isoformat() for t in f["ds"]],
        "value": [float(v) for v in f["y"]],
    }


def _dial_json(name: str, reading: DialReading | None, alert: Any, reason: str) -> dict[str, Any]:
    assess = {a.dial: a for a in alert.dials} if alert is not None else {}
    a = assess.get(name)
    out: dict[str, Any] = {
        "dial": name,
        "symbol": _SYMBOLS[name],
        "unit": _UNITS[name],
        "state": "danger" if a is not None and a.in_danger else "calm",
        "p_danger": float(a.p_danger) if a is not None else 0.0,
    }
    if reading is None:
        return out | {
            "value": 0.0,
            "available": False,
            "note": reason,
            "history": {"t": [], "value": []},
        }
    hist = reading.detail.get("history")
    if not isinstance(hist, dict):
        hist = {"t": [reading.as_of.isoformat()], "value": [_finite(reading.value)]}
    return out | {
        "value": _finite(reading.value),
        "level": reading.level,
        "ci": [_finite(reading.ci_lo), _finite(reading.ci_hi)],
        "as_of": reading.as_of.isoformat(),
        "history": hist,
    }


def _acceleration_json(ctx: RunContext, readings: Mapping[str, DialReading]) -> list[Any]:
    fits = ctx.artifacts.get("fit", {}).get("fits", {})
    if LEDGER_SERIES not in fits:
        return []
    f = fits[LEDGER_SERIES]
    hist = f.history.dropna(subset=["y"])
    fc = f.forecast.set_index("ds").reindex(hist["ds"])
    acc = readings.get("acceleration")
    samples = acc.detail.get("delta_samples_" + LEDGER_SERIES, []) if acc is not None else []
    return [
        {
            "series": LEDGER_SERIES,
            "unit": "",
            "t": [pd.Timestamp(t).date().isoformat() for t in hist["ds"]],
            "observed": [float(v) for v in hist["y"]],
            "fit": [_finite(float(v)) for v in fc["yhat"]],
            "lower": [_finite(float(v)) for v in fc["yhat_lower"]],
            "upper": [_finite(float(v)) for v in fc["yhat_upper"]],
            "changepoints": [pd.Timestamp(c).date().isoformat() for c in f.changepoints],
            "delta_samples": [float(s) for s in samples],
        }
    ]


def _ledger_json(ctx: RunContext) -> list[Any]:
    from kairos.ledger.chain import Ledger
    from kairos.ledger.timestamp import ProofStatus, check_ots_proof, proof_path

    book = ctx.artifacts.get("ledger", {}).get("ledger")
    if book is None:
        book = Ledger(ctx.root / "ledger" / "forecasts.jsonl")
    if not book.path.exists():
        return []
    proofs_dir = book.path.parent / "proofs"
    out: list[Any] = []
    for e in book.forecasts():
        path = proof_path(e, proofs_dir)
        status = ProofStatus.MISSING
        if path.exists():
            status = check_ots_proof(bytes.fromhex(e.entry_hash), path.read_bytes()).status
        proof = {"anchored": "verified", "invalid": "failed"}.get(str(status), "pending")
        p = e.payload.get("forecast", {})
        points = p.get("points", [])
        out.append(
            {
                "id": str(e.seq),
                "registered_at": e.recorded_at.isoformat(),
                "statement": f"{p.get('series_id')} {p.get('model')} forecast from "
                f"{p.get('as_of')} ({len(points)} months, 90% band)",
                "hash": e.entry_hash,
                "prev_hash": e.prev_hash,
                "proof": proof,
                "outcome_date": points[-1]["ds"] if points else str(p.get("as_of", "")),
                "score": None,
            }
        )
    return out


def build_snapshot(ctx: RunContext) -> dict[str, Any]:
    """Assemble the schema-v1 dashboard snapshot from this run's artifacts."""
    from kairos.dashboard.snapshot import SCHEMA_VERSION
    from kairos.data.manifest import read_manifest
    from kairos.vinge.dial import load_thresholds

    datasets = _load(ctx)
    readings = _readings(ctx)
    skipped: Mapping[str, str] = ctx.artifacts.get("dials", {}).get("skipped", {})
    alert = ctx.artifacts.get("governor", {}).get("alert")
    vintages = [d.as_of for d in readings.values()]
    try:
        vintage = str(read_manifest(_data_root(ctx)).get("vintage", ""))
    except (FileNotFoundError, ValueError):
        vintage = ""
    if not vintage:
        vintage = max(vintages).isoformat() if vintages else ctx.as_of.date().isoformat()
    vinge = readings.get("vinge")
    vh = vinge.detail.get("history", {"t": [], "value": []}) if vinge is not None else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": ctx.as_of.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_vintage": vintage,
        "is_demo": False,
        "run_id": ctx.run_id,
        "parameters": {
            "lambda": float(alert.lam) if alert is not None else float("nan"),
            "tripwires_version": str(alert.spec_version) if alert is not None else "",
            "vinge_tolerance_factor": float(load_thresholds().tolerance_factor),
        },
        "alert": {
            "level": alert.level if alert is not None else 0,
            "name": alert.name if alert is not None else None,
            "response": alert.response if alert is not None else None,
            "lead_time": None,
        },
        "dials": [
            _dial_json(n, readings.get(n), alert, skipped.get(n, "not computed"))
            for n in DIAL_NAMES
        ],
        "capability": _series_json(datasets, "idx.C", "C(t)"),
        "human_capacity": _series_json(datasets, "idx.H", "H(t)"),
        "acceleration": _acceleration_json(ctx, readings),
        "vinge_horizon": {
            "t": list(vh.get("t", [])),
            "h_star": list(vh.get("value", [])),
            "unit": "months",
        },
        "ledger": _ledger_json(ctx),
    }


def export(ctx: RunContext) -> Mapping[str, Any]:
    """Write ``build/snapshot.json`` and render the static site to ``build/site``."""
    from kairos.dashboard.export import export_site
    from kairos.dashboard.snapshot import parse_snapshot

    snap = build_snapshot(ctx)
    if not math.isfinite(snap["parameters"]["lambda"]):
        from kairos.philosophy.jonas import load_policy

        snap["parameters"]["lambda"] = float(load_policy().jonas.lam)
    parse_snapshot(snap)
    build = ctx.root / "build"
    build.mkdir(parents=True, exist_ok=True)
    path = build / "snapshot.json"
    path.write_text(json.dumps(snap, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    index = export_site(path, build / "site")
    return {"detail": f"{path.relative_to(ctx.root)} -> {index.relative_to(ctx.root)}"}
