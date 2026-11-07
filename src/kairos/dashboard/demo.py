"""Synthetic demo snapshot.

Real snapshots come from the weekly pipeline. Until those phases land, this
module builds a clearly labelled fake one (``is_demo: true``) so the app and
the static export can be rendered and tested. Nothing here is a forecast.

Regenerate the fixture with ``python -m kairos.dashboard.demo``.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from kairos.dashboard.snapshot import DEMO_SNAPSHOT_PATH, SCHEMA_VERSION

_SEED = 20261106


def _months(start_year: int, n: int) -> list[str]:
    return [f"{start_year + (i // 12)}-{(i % 12) + 1:02d}-01" for i in range(n)]


def _r(x: float) -> float:
    return round(x, 4)


def build_demo_snapshot() -> dict[str, object]:
    """Return the synthetic demo snapshot as a JSON-ready dict."""
    rng = random.Random(_SEED)
    n = 202  # 2010-01 .. 2026-10
    t = _months(2010, n)
    years = [i / 12 for i in range(n)]

    # C(t): mildly super-exponential; H(t): slower exponential. Both log10 of
    # an index normalised to 1 at the base year (D8).
    cap = [_r(0.25 * y + 0.004 * y * y + rng.gauss(0, 0.04)) for y in years]
    hum = [_r(0.08 * y + rng.gauss(0, 0.03)) for y in years]

    def history(f: list[float]) -> dict[str, object]:
        """Last three years of a series, as a dial history."""
        return {"t": t[-36:], "value": [_r(v) for v in f[-36:]]}

    delta_hist = [0.02 + 0.0006 * i + rng.gauss(0, 0.01) for i in range(n)]
    kappa_hist = [0.05 + 0.0003 * i + rng.gauss(0, 0.01) for i in range(n)]
    h_star = [max(1.0, 30.0 - 0.09 * i + rng.gauss(0, 1.2)) for i in range(n)]
    gap_rate = [0.15 + 0.0005 * i + rng.gauss(0, 0.02) for i in range(n)]

    dials = [
        {
            "dial": "acceleration",
            "symbol": "δ",
            "value": _r(delta_hist[-1]),
            "unit": "",
            "state": "danger",
            "p_danger": 0.81,
            "history": history(delta_hist),
        },
        {
            "dial": "recursion",
            "symbol": "κ",
            "value": _r(kappa_hist[-1]),
            "unit": "",
            "state": "calm",
            "p_danger": 0.22,
            "history": history(kappa_hist),
        },
        {
            "dial": "vinge_horizon",
            "symbol": "h*",
            "value": _r(h_star[-1]),
            "unit": "months",
            "state": "calm",
            "p_danger": 0.34,
            "history": history(h_star),
        },
        {
            "dial": "jonas_gap",
            "symbol": "dG/dt",
            "value": _r(gap_rate[-1]),
            "unit": "per year",
            "state": "danger",
            "p_danger": 0.93,
            "history": history(gap_rate),
        },
    ]

    accel = []
    for name, unit, slope, curve in (
        ("training_compute", "log10 FLOP", 0.25, 0.004),
        ("metr_time_horizon", "log10 minutes", 0.18, 0.006),
    ):
        obs = [slope * y + curve * y * y + rng.gauss(0, 0.05) for y in years]
        fit = [slope * y + curve * y * y for y in years]
        band = [0.06 + 0.002 * y for y in years]
        center = 0.08 + 0.02 * rng.random()
        accel.append(
            {
                "series": name,
                "unit": unit,
                "t": t,
                "observed": [_r(v) for v in obs],
                "fit": [_r(v) for v in fit],
                "lower": [_r(f - b) for f, b in zip(fit, band, strict=True)],
                "upper": [_r(f + b) for f, b in zip(fit, band, strict=True)],
                "changepoints": [t[i] for i in (48, 96, 140, 170)],
                "delta_samples": [_r(rng.gauss(center, 0.035)) for _ in range(400)],
            }
        )

    ledger: list[dict[str, object]] = []
    prev: str | None = None
    for k in range(8):
        reg = t[n - 40 + 4 * k]
        statement = (
            f"If no new intervention, training compute reaches 10^{27 + k * 0.25:.2f} FLOP "
            f"by {int(reg[:4]) + 2}{reg[4:]} (90% interval)."
        )
        body = {
            "id": f"F-{k + 1:04d}",
            "registered_at": f"{reg}T00:00:00Z",
            "statement": statement,
            "prev_hash": prev,
        }
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        matured = k < 3
        ledger.append(
            {
                **body,
                "hash": digest,
                "proof": "verified" if k < 6 else "pending",
                "outcome_date": f"{int(reg[:4]) + 2}{reg[4:]}" if not matured else t[n - 30 + k],
                "score": _r(0.1 + 0.05 * k) if matured else None,
            }
        )
        prev = digest

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-11-06T00:00:00Z",
        "data_vintage": t[-1],
        "is_demo": True,
        "parameters": {
            "lambda": 9.0,
            "tripwires_version": "0.0.0-demo",
            "vinge_tolerance_factor": 2.0,
        },
        "alert": {
            "level": 2,
            "lead_time": {
                "median": 31.0,
                "lower": 18.0,
                "upper": 52.0,
                "interval": 0.9,
                "unit": "months",
            },
        },
        "dials": dials,
        "capability": {
            "name": "C(t) capability index",
            "unit": "log10 index (base 2010 = 0)",
            "t": t,
            "value": cap,
        },
        "human_capacity": {
            "name": "H(t) human capacity index",
            "unit": "log10 index (base 2010 = 0)",
            "t": t,
            "value": hum,
        },
        "acceleration": accel,
        "vinge_horizon": {
            "t": t,
            "h_star": [_r(v) for v in h_star],
            "unit": "months",
            "by_tolerance": {
                "1.5": [_r(0.7 * v) for v in h_star],
                "3.0": [_r(1.4 * v) for v in h_star],
            },
        },
        "ledger": ledger,
    }


def write_demo_snapshot(path: Path = DEMO_SNAPSHOT_PATH) -> Path:
    """Write the demo snapshot fixture and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build_demo_snapshot(), ensure_ascii=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_demo_snapshot())
