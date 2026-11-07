"""Ledger browser logic: rows for display and a link check.

This only checks that each published entry points at the one before it. Full
verification (recomputing hashes and checking timestamp proofs) is the job of
``kairos ledger verify`` (K-10.06); the dashboard displays its result.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from kairos.dashboard.snapshot import LedgerEntry, ProofStatus

PROOF_LABEL: dict[ProofStatus, str] = {
    "verified": "✔ verified",
    "pending": "… pending",
    "failed": "✖ failed",
}


def short_hash(h: str, n: int = 12) -> str:
    """Abbreviate a hex digest for display."""
    return h[:n] + "…" if len(h) > n else h


def ledger_rows(entries: Iterable[LedgerEntry]) -> list[dict[str, object]]:
    """One display row per entry, newest first."""
    rows: list[dict[str, object]] = [
        {
            "ID": e.id,
            "Registered": e.registered_at,
            "Forecast": e.statement,
            "Outcome date": e.outcome_date,
            "Proof": PROOF_LABEL[e.proof],
            "Score": e.score,
            "Hash": short_hash(e.hash),
            "Full hash": e.hash,
        }
        for e in entries
    ]
    return sorted(rows, key=lambda r: str(r["Registered"]), reverse=True)


def chain_breaks(entries: Sequence[LedgerEntry]) -> list[str]:
    """IDs of entries whose ``prev_hash`` does not match the entry before them."""
    breaks = []
    for i, e in enumerate(entries):
        expected = entries[i - 1].hash if i else None
        if e.prev_hash != expected:
            breaks.append(e.id)
    return breaks


def proof_counts(entries: Iterable[LedgerEntry]) -> dict[ProofStatus, int]:
    """Count entries by proof status."""
    counts: dict[ProofStatus, int] = {"verified": 0, "pending": 0, "failed": 0}
    for e in entries:
        counts[e.proof] += 1
    return counts
