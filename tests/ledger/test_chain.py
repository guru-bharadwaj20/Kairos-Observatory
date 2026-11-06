import json
from datetime import UTC, datetime, timedelta

import pytest

from kairos.ledger.chain import (
    GENESIS_HASH,
    Ledger,
    LedgerEntry,
    LedgerError,
    LedgerIntegrityError,
    manifest_hash,
)
from kairos.ledger.forecast import forecast_hash

MANIFEST = "a" * 64
T0 = datetime(2026, 11, 6, 12, tzinfo=UTC)


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger" / "forecasts.jsonl")


def test_empty_ledger(ledger):
    assert ledger.entries() == []
    assert ledger.head() is None


def test_append_chains_entries(ledger, make_forecast):
    e0 = ledger.append_forecast(make_forecast(), manifest_hash=MANIFEST, recorded_at=T0)
    e1 = ledger.append_forecast(
        make_forecast(base=200.0), manifest_hash=MANIFEST, recorded_at=T0 + timedelta(days=7)
    )
    assert e0.seq == 0
    assert e0.prev_hash == GENESIS_HASH
    assert e1.prev_hash == e0.entry_hash
    assert e0.manifest_hash == MANIFEST
    assert ledger.entries() == [e0, e1]
    assert ledger.head() == e1


def test_lines_are_canonical_and_lf(ledger, make_forecast):
    entry = ledger.append_forecast(make_forecast(), manifest_hash=MANIFEST, recorded_at=T0)
    raw = ledger.path.read_bytes()
    assert raw == entry.to_line()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 1


def test_forecast_round_trips_with_same_hash(ledger, make_forecast):
    f = make_forecast()
    ledger.append_forecast(f, manifest_hash=MANIFEST, recorded_at=T0)
    (entry,) = Ledger(ledger.path).entries()
    assert entry.payload["forecast_hash"] == forecast_hash(f)
    assert forecast_hash(entry.forecast()) == forecast_hash(f)


def test_duplicate_forecast_rejected(ledger, make_forecast):
    ledger.append_forecast(make_forecast(), manifest_hash=MANIFEST, recorded_at=T0)
    with pytest.raises(LedgerError, match="already registered"):
        ledger.append_forecast(make_forecast(), manifest_hash=MANIFEST)


def test_time_cannot_go_backwards(ledger, make_forecast):
    ledger.append_forecast(make_forecast(), manifest_hash=MANIFEST, recorded_at=T0)
    with pytest.raises(LedgerError, match="backwards"):
        ledger.append("intervention", {"name": "x"}, recorded_at=T0 - timedelta(days=1))


def test_correction_appends_never_edits(ledger, make_forecast):
    e0 = ledger.append_forecast(make_forecast(), manifest_hash=MANIFEST, recorded_at=T0)
    before = ledger.path.read_bytes()
    c = ledger.append_correction(e0.entry_hash, "wrong unit")
    assert ledger.path.read_bytes().startswith(before)
    assert c.kind == "correction"
    assert c.payload["supersedes"] == e0.entry_hash
    with pytest.raises(KeyError):
        ledger.append_correction("b" * 64, "no such entry")


def test_append_refuses_tampered_ledger(ledger, make_forecast):
    ledger.append_forecast(make_forecast(), manifest_hash=MANIFEST, recorded_at=T0)
    text = ledger.path.read_text().replace('"frontier_compute"', '"other"')
    ledger.path.write_text(text, newline="\n")
    with pytest.raises(LedgerIntegrityError, match="payload"):
        ledger.append("intervention", {"name": "x"})


def test_header_tamper_detected(ledger, make_forecast):
    ledger.append_forecast(make_forecast(), manifest_hash=MANIFEST, recorded_at=T0)
    record = json.loads(ledger.path.read_text())
    record["manifest_hash"] = "c" * 64
    ledger.path.write_text(json.dumps(record) + "\n", newline="\n")
    with pytest.raises(LedgerIntegrityError, match="entry_hash"):
        ledger.entries()


def test_garbage_and_blank_lines_rejected(ledger):
    ledger.path.parent.mkdir(parents=True)
    ledger.path.write_text("not json\n", newline="\n")
    with pytest.raises(LedgerIntegrityError, match="line 1"):
        ledger.entries()
    ledger.path.write_text("\n", newline="\n")
    with pytest.raises(LedgerIntegrityError, match="blank"):
        ledger.entries()


def test_lock_blocks_concurrent_append(ledger):
    ledger.path.parent.mkdir(parents=True)
    ledger.path.with_name("forecasts.jsonl.lock").touch()
    with pytest.raises(LedgerError, match="another process"):
        ledger.append("intervention", {"name": "x"})


def test_manifest_hash_ignores_formatting(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"x": 1, "y": [1, 2]}')
    b.write_text('{\n  "y": [1,2],\n  "x": 1\n}')
    assert manifest_hash(a) == manifest_hash(b)
    with pytest.raises(FileNotFoundError):
        manifest_hash(tmp_path / "missing.json")


def test_entry_rejects_bad_hashes():
    with pytest.raises(ValueError, match="SHA-256"):
        LedgerEntry.create(seq=0, kind="score", payload={}, prev_hash="zz")


def test_naive_recorded_at_rejected(ledger):
    with pytest.raises(ValueError, match="timezone-aware"):
        ledger.append("score", {}, recorded_at=datetime(2026, 1, 1))
