"""Tamper tests for ``kairos ledger verify``. Calendars are local fakes."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from kairos.cli import app
from kairos.ledger.chain import Ledger, LedgerIntegrityError
from kairos.ledger.timestamp import ProofStatus, proof_path, stamp_entries
from kairos.ledger.verify import verify_ledger

T0 = datetime(2026, 11, 6, 12, tzinfo=UTC)
MANIFEST = "a" * 64
runner = CliRunner()


@pytest.fixture
def fake_stamper():
    pytest.importorskip("opentimestamps")
    from opentimestamps.core.notary import PendingAttestation
    from opentimestamps.core.timestamp import Timestamp

    from kairos.ledger.timestamp import OtsTimestamper

    class Cal:
        def __init__(self, url):
            self.url = url

        def submit(self, digest, timeout=None):
            t = Timestamp(digest)
            t.attestations.add(PendingAttestation(self.url))
            return t

    return OtsTimestamper(("https://a.test", "https://b.test"), calendar_factory=Cal)


@pytest.fixture
def built(tmp_path, make_forecast, fake_stamper):
    ledger = Ledger(tmp_path / "forecasts.jsonl")
    for i in range(4):
        ledger.append_forecast(
            make_forecast(base=100.0 + i),
            manifest_hash=MANIFEST,
            recorded_at=T0 + timedelta(weeks=i),
        )
    proofs = tmp_path / "proofs"
    stamp_entries(ledger.entries(), fake_stamper, proofs)
    return ledger, proofs


def lines(ledger):
    return ledger.path.read_text().splitlines(keepends=True)


def write(ledger, rows):
    ledger.path.write_text("".join(rows), newline="\n")


def test_clean_ledger_verifies(built):
    ledger, proofs = built
    report = verify_ledger(ledger.path, proofs, require_proofs=True)
    assert report.ok, report.issues
    assert report.n_entries == 4
    assert set(report.proofs.values()) == {ProofStatus.PENDING}
    report.raise_if_failed()


def test_edited_payload_fails(built):
    ledger, proofs = built
    rows = lines(ledger)
    rows[1] = rows[1].replace('"frontier_compute"', '"frontier_compuTe"')
    write(ledger, rows)
    report = verify_ledger(ledger.path, proofs)
    assert not report.ok
    assert any("payload_hash mismatch" in i.message for i in report.errors)
    with pytest.raises(LedgerIntegrityError, match="payload_hash"):
        report.raise_if_failed()


def test_edited_and_rehashed_entry_fails(built):
    """Recomputing the edited entry's own hashes still breaks the next link and its proof."""
    from kairos.ledger.chain import LedgerEntry

    ledger, proofs = built
    rows = lines(ledger)
    old = LedgerEntry.model_validate(json.loads(rows[1]))
    payload = dict(old.payload)
    payload["forecast_hash"] = "f" * 64
    forged = LedgerEntry.create(
        seq=1,
        kind="forecast",
        payload=payload,
        prev_hash=old.prev_hash,
        manifest_hash=old.manifest_hash,
        recorded_at=old.recorded_at,
    )
    rows[1] = forged.to_line().decode()
    write(ledger, rows)
    messages = [i.message for i in verify_ledger(ledger.path, proofs).errors]
    assert any("entry 2: prev_hash" in m for m in messages)
    assert any("forecast_hash mismatch" in m for m in messages)
    assert any("orphan proof 000001" in m for m in messages)


def test_deleted_middle_entry_fails(built):
    ledger, proofs = built
    rows = lines(ledger)
    del rows[1]
    write(ledger, rows)
    messages = [i.message for i in verify_ledger(ledger.path, proofs).errors]
    assert any("seq is 2" in m for m in messages)
    assert any("prev_hash" in m for m in messages)
    assert any("orphan proof" in m for m in messages)


def test_deleted_last_entry_fails_via_orphan_proof(built):
    ledger, proofs = built
    rows = lines(ledger)
    head = json.loads(rows[-1])["entry_hash"]
    write(ledger, rows[:-1])
    report = verify_ledger(ledger.path, proofs)
    assert any("orphan proof 000003" in i.message for i in report.errors)
    # even without proofs, a known head catches truncation
    report = verify_ledger(ledger.path, proofs.parent / "none", expect_head=head)
    assert any("expected head" in i.message for i in report.errors)


def test_reordered_entries_fail(built):
    ledger, proofs = built
    rows = lines(ledger)
    rows[1], rows[2] = rows[2], rows[1]
    write(ledger, rows)
    messages = [i.message for i in verify_ledger(ledger.path, proofs).errors]
    assert any("seq is 2" in m for m in messages)
    assert any("seq is 1" in m for m in messages)


def test_reserialised_line_fails(built):
    ledger, proofs = built
    rows = lines(ledger)
    rows[0] = json.dumps(json.loads(rows[0]), indent=None) + "\n"
    write(ledger, rows)
    messages = [i.message for i in verify_ledger(ledger.path, proofs).errors]
    assert any("canonical form" in m for m in messages)


def test_crlf_and_garbage_fail(built):
    ledger, proofs = built
    ledger.path.write_bytes(ledger.path.read_bytes().replace(b"\n", b"\r\n"))
    assert any("CR" in i.message for i in verify_ledger(ledger.path, proofs).errors)
    write(ledger, ["{not json}\n"])
    assert any("line 1" in i.message for i in verify_ledger(ledger.path, proofs).errors)


def test_tampered_or_swapped_proof_fails(built):
    ledger, proofs = built
    entries = ledger.entries()
    p0, p1 = proof_path(entries[0], proofs), proof_path(entries[1], proofs)
    p0.write_bytes(p1.read_bytes())
    report = verify_ledger(ledger.path, proofs)
    assert report.proofs[0] is ProofStatus.INVALID
    assert any("different hash" in i.message for i in report.errors)


def test_missing_proof_is_warning_unless_required(built):
    ledger, proofs = built
    proof_path(ledger.entries()[2], proofs).unlink()
    assert verify_ledger(ledger.path, proofs).ok
    assert not verify_ledger(ledger.path, proofs, require_proofs=True).ok


def test_cli_verify_ok_and_fail(built):
    ledger, proofs = built
    args = ["ledger", "verify", "--ledger", str(ledger.path), "--proofs", str(proofs)]
    ok = runner.invoke(app, args)
    assert ok.exit_code == 0, ok.output
    assert "OK" in ok.output
    rows = lines(ledger)
    rows[0] = rows[0].replace('"FLOP"', '"flop"')
    write(ledger, rows)
    bad = runner.invoke(app, [*args, "--json"])
    assert bad.exit_code == 1
    assert json.loads(bad.output)["ok"] is False


def test_cli_show_and_stamp_offline(built, monkeypatch):
    from kairos.commands import ledger as ledger_cmd

    ledger, proofs = built
    shown = runner.invoke(
        app, ["ledger", "show", "--ledger", str(ledger.path), "--proofs", str(proofs)]
    )
    assert shown.exit_code == 0, shown.output
    assert "forecast" in shown.output
    monkeypatch.setattr(ledger_cmd, "default_timestamper", lambda: None)
    new_proofs = proofs.parent / "fresh"
    stamped = runner.invoke(
        app, ["ledger", "stamp", "--ledger", str(ledger.path), "--proofs", str(new_proofs)]
    )
    assert stamped.exit_code == 0
    assert "missing" in stamped.output
    upgraded = runner.invoke(app, ["ledger", "upgrade", "--ledger", str(ledger.path)])
    assert upgraded.exit_code == 1
