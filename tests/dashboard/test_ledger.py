from __future__ import annotations

from dataclasses import replace

from kairos.dashboard import ledger_view
from kairos.dashboard.snapshot import load_demo


def test_demo_chain_is_intact() -> None:
    assert ledger_view.chain_breaks(load_demo().ledger) == []


def test_detects_edited_and_reordered_entries() -> None:
    entries = list(load_demo().ledger)
    edited = [*entries[:2], replace(entries[2], prev_hash="0" * 64), *entries[3:]]
    assert ledger_view.chain_breaks(edited) == [entries[2].id]
    swapped = [entries[1], entries[0], *entries[2:]]
    assert entries[1].id in ledger_view.chain_breaks(swapped)


def test_rows_show_every_entry_newest_first() -> None:
    entries = load_demo().ledger
    rows = ledger_view.ledger_rows(entries)
    assert len(rows) == len(entries)
    assert rows[0]["ID"] == entries[-1].id
    assert rows[0]["Full hash"] == entries[-1].hash
    assert str(rows[0]["Hash"]).endswith("…")
    assert {r["Proof"] for r in rows} <= set(ledger_view.PROOF_LABEL.values())


def test_proof_counts() -> None:
    counts = ledger_view.proof_counts(load_demo().ledger)
    assert sum(counts.values()) == len(load_demo().ledger)
    assert counts["pending"] > 0
