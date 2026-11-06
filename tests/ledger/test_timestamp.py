"""OpenTimestamps tests. Every calendar is a local fake: no network is used."""

from datetime import UTC, datetime
from typing import ClassVar

import pytest

from kairos.ledger import timestamp as ts
from kairos.ledger.chain import Ledger
from kairos.ledger.timestamp import (
    OtsTimestamper,
    ProofStatus,
    TimestampUnavailableError,
    check_ots_proof,
    proof_path,
    stamp_entries,
)

pytest.importorskip("opentimestamps")

from opentimestamps.core.notary import (
    BitcoinBlockHeaderAttestation,
    PendingAttestation,
)
from opentimestamps.core.timestamp import Timestamp

T0 = datetime(2026, 11, 6, 12, tzinfo=UTC)


class FakeCalendar:
    """Answers like a calendar server; ``anchored`` makes upgrades succeed."""

    submitted: ClassVar[list[bytes]] = []

    def __init__(self, url, *, fail=False, anchored=False):
        self.url = url
        self.fail = fail
        self.anchored = anchored

    def submit(self, digest, timeout=None):
        if self.fail:
            raise OSError("offline")
        FakeCalendar.submitted.append(digest)
        stamp = Timestamp(digest)
        stamp.attestations.add(PendingAttestation(self.url))
        return stamp

    def get_timestamp(self, commitment, timeout=None):
        if not self.anchored:
            raise KeyError("Pending confirmation in Bitcoin blockchain")
        stamp = Timestamp(commitment)
        stamp.attestations.add(BitcoinBlockHeaderAttestation(900_000))
        return stamp


def factory(*, fail=(), anchored=False):
    return lambda url: FakeCalendar(url, fail=url in fail, anchored=anchored)


CALS = ("https://cal-a.test", "https://cal-b.test", "https://cal-c.test")
DIGEST = bytes.fromhex("ab" * 32)


@pytest.fixture
def stamper():
    return OtsTimestamper(CALS, calendar_factory=factory())


def test_stamp_produces_pending_proof(stamper):
    proof = stamper.stamp(DIGEST)
    assert proof.startswith(b"\x00OpenTimestamps\x00\x00Proof\x00")
    check = stamper.check(DIGEST, proof)
    assert check.status is ProofStatus.PENDING
    assert set(check.attestations) == set(CALS)


def test_nonce_hides_digest(stamper):
    FakeCalendar.submitted.clear()
    stamper.stamp(DIGEST)
    assert FakeCalendar.submitted
    assert DIGEST not in FakeCalendar.submitted


def test_wrong_digest_and_garbage_invalid(stamper):
    proof = stamper.stamp(DIGEST)
    assert check_ots_proof(b"\x00" * 32, proof).status is ProofStatus.INVALID
    assert check_ots_proof(DIGEST, b"garbage").status is ProofStatus.INVALID
    assert check_ots_proof(DIGEST, proof[:-3]).status is ProofStatus.INVALID


def test_too_few_calendars(stamper):
    s = OtsTimestamper(CALS, calendar_factory=factory(fail=CALS[:2]))
    with pytest.raises(TimestampUnavailableError, match="1 of 2"):
        s.stamp(DIGEST)
    one_down = OtsTimestamper(CALS, calendar_factory=factory(fail=CALS[:1]))
    assert one_down.check(DIGEST, one_down.stamp(DIGEST)).status is ProofStatus.PENDING


def test_upgrade_to_anchored(stamper):
    proof = stamper.stamp(DIGEST)
    still_pending = stamper.upgrade(DIGEST, proof)
    assert stamper.check(DIGEST, still_pending).status is ProofStatus.PENDING
    up = OtsTimestamper(CALS, calendar_factory=factory(anchored=True)).upgrade(DIGEST, proof)
    check = stamper.check(DIGEST, up)
    assert check.status is ProofStatus.ANCHORED
    assert "bitcoin:900000" in check.attestations
    with pytest.raises(ValueError, match="commit"):
        stamper.upgrade(b"\x01" * 32, proof)


def test_digest_length_checked(stamper):
    with pytest.raises(ValueError, match="32 bytes"):
        stamper.stamp(b"short")


def test_stamp_entries_writes_proof_files(tmp_path, stamper):
    ledger = Ledger(tmp_path / "forecasts.jsonl")
    e0 = ledger.append("intervention", {"name": "a"}, recorded_at=T0)
    e1 = ledger.append("intervention", {"name": "b"}, recorded_at=T0)
    proofs = tmp_path / "proofs"
    result = stamp_entries(ledger.entries(), stamper, proofs)
    assert result == {0: ProofStatus.PENDING, 1: ProofStatus.PENDING}
    assert proof_path(e0, proofs).name == f"000000-{e0.entry_hash[:16]}.ots"
    assert proof_path(e1, proofs).exists()
    before = proof_path(e0, proofs).read_bytes()
    stamp_entries(ledger.entries(), stamper, proofs)  # idempotent: existing proofs kept
    assert proof_path(e0, proofs).read_bytes() == before


def test_stamp_entries_degrades_when_offline(tmp_path, caplog):
    ledger = Ledger(tmp_path / "forecasts.jsonl")
    ledger.append("intervention", {"name": "a"}, recorded_at=T0)
    offline = OtsTimestamper(CALS, calendar_factory=factory(fail=CALS))
    result = stamp_entries(ledger.entries(), offline, tmp_path / "proofs")
    assert result == {0: ProofStatus.MISSING}
    assert "not timestamped" in caplog.text
    assert stamp_entries(ledger.entries(), None, tmp_path / "proofs") == {0: ProofStatus.MISSING}


def test_without_opentimestamps(monkeypatch):
    monkeypatch.setattr(ts, "ots_available", lambda: False)
    with pytest.raises(TimestampUnavailableError, match="--extra ots"):
        OtsTimestamper(CALS)
    assert ts.default_timestamper() is None
    assert check_ots_proof(DIGEST, b"x").status is ProofStatus.UNAVAILABLE
