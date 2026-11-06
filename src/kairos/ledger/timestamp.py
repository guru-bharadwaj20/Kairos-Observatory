"""External timestamping of ledger entries with OpenTimestamps (D11).

Git history can be rewritten, so the ledger's own hashes only prove *order*, not
*time*. Each entry's ``entry_hash`` is therefore submitted to public
OpenTimestamps calendars, which aggregate it into a Bitcoin transaction. The
resulting proof (a standard ``.ots`` file, readable by ``ots info`` /
``ots verify``) is committed under ``ledger/proofs/``.

The OpenTimestamps library is optional (``uv sync --extra ots``). Without it, or
without network, stamping is skipped with a warning and verification reports
proofs as ``unavailable``; nothing else in the ledger needs it. Only the
``opentimestamps`` core package is used, not ``otsclient`` (whose
``python-bitcoinlib`` dependency fails to import on Windows).

Proof file layout: ``ledger/proofs/<seq:06d>-<entry_hash[:16]>.ots``. The proof's
*file digest* is the 32-byte ``entry_hash`` itself (hash op ``sha256``), so
``ots verify -d <entry_hash> <proof>`` checks it with the reference client.
"""

from __future__ import annotations

import enum
import importlib.util
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from kairos.ledger.chain import LedgerEntry

__all__ = [
    "DEFAULT_CALENDARS",
    "DEFAULT_PROOFS_DIR",
    "OtsTimestamper",
    "ProofCheck",
    "ProofStatus",
    "TimestampUnavailableError",
    "Timestamper",
    "check_ots_proof",
    "default_timestamper",
    "ots_available",
    "proof_path",
    "stamp_entries",
]

log = logging.getLogger(__name__)

DEFAULT_PROOFS_DIR: Final = Path("ledger/proofs")

DEFAULT_CALENDARS: Final = (
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
    "https://ots.btc.catallaxy.com",
)
"""The public calendars the reference ``ots`` client uses by default."""


class TimestampUnavailableError(RuntimeError):
    """OpenTimestamps is not installed, or no calendar could be reached."""


class ProofStatus(enum.StrEnum):
    """Outcome of checking one proof."""

    MISSING = "missing"
    """No proof file for the entry."""
    INVALID = "invalid"
    """The proof is corrupt or commits to a different hash."""
    PENDING = "pending"
    """Calendars promised to anchor it; not yet in a Bitcoin block (run ``upgrade``)."""
    ANCHORED = "anchored"
    """Contains a Bitcoin block-header attestation."""
    UNAVAILABLE = "unavailable"
    """OpenTimestamps is not installed, so the proof could not be parsed."""


@dataclass(frozen=True)
class ProofCheck:
    """Result of checking one proof.

    Attributes:
        status: See :class:`ProofStatus`.
        detail: Human-readable explanation.
        attestations: Calendar URLs (pending) and ``bitcoin:<height>`` (anchored).
    """

    status: ProofStatus
    detail: str = ""
    attestations: tuple[str, ...] = ()


class Timestamper(Protocol):
    """Creates and checks timestamp proofs for 32-byte digests."""

    def stamp(self, digest: bytes) -> bytes:
        """Return proof bytes for ``digest``."""
        ...

    def upgrade(self, digest: bytes, proof: bytes) -> bytes:
        """Return ``proof`` with pending attestations completed where possible."""
        ...

    def check(self, digest: bytes, proof: bytes) -> ProofCheck:
        """Check ``proof`` against ``digest`` without network access."""
        ...


def ots_available() -> bool:
    """Return whether the ``opentimestamps`` package can be imported."""
    return importlib.util.find_spec("opentimestamps") is not None


def proof_path(entry: LedgerEntry, proofs_dir: Path = DEFAULT_PROOFS_DIR) -> Path:
    """Return where the proof for ``entry`` lives."""
    return Path(proofs_dir) / f"{entry.seq:06d}-{entry.entry_hash[:16]}.ots"


class CalendarClient(Protocol):
    """The subset of :class:`opentimestamps.calendar.RemoteCalendar` that is used."""

    url: str

    def submit(self, digest: bytes, timeout: float | None = None) -> Any:
        """Submit ``digest``; return an ``opentimestamps`` ``Timestamp``."""
        ...

    def get_timestamp(self, commitment: bytes, timeout: float | None = None) -> Any:
        """Return the upgraded ``Timestamp`` for ``commitment``."""
        ...


def _remote_calendar(url: str) -> CalendarClient:
    from opentimestamps.calendar import RemoteCalendar

    cal: CalendarClient = RemoteCalendar(url, user_agent="kairos-observatory")
    return cal


class OtsTimestamper:
    """OpenTimestamps-backed :class:`Timestamper`.

    Args:
        calendars: Calendar URLs to submit to.
        min_calendars: Stamping fails unless at least this many calendars answer.
        timeout: Per-request timeout in seconds.
        calendar_factory: Builds a client for a URL; injectable for tests.

    Raises:
        TimestampUnavailableError: If ``opentimestamps`` is not installed.
    """

    def __init__(
        self,
        calendars: Sequence[str] = DEFAULT_CALENDARS,
        *,
        min_calendars: int = 2,
        timeout: float = 10.0,
        calendar_factory: Callable[[str], CalendarClient] = _remote_calendar,
    ) -> None:
        if not ots_available():
            raise TimestampUnavailableError(
                "opentimestamps is not installed; run `uv sync --extra ots`"
            )
        if not calendars:
            raise ValueError("need at least one calendar")
        self.calendars = tuple(calendars)
        self.min_calendars = min(min_calendars, len(self.calendars))
        self.timeout = timeout
        self._factory = calendar_factory

    # serialisation -------------------------------------------------------
    @staticmethod
    def _dump(detached: Any) -> bytes:
        from opentimestamps.core.serialize import BytesSerializationContext

        ctx = BytesSerializationContext()
        detached.serialize(ctx)
        data: bytes = ctx.getbytes()
        return data

    @staticmethod
    def _load(proof: bytes) -> Any:
        from opentimestamps.core.serialize import BytesDeserializationContext
        from opentimestamps.core.timestamp import DetachedTimestampFile

        return DetachedTimestampFile.deserialize(BytesDeserializationContext(proof))

    # Timestamper ---------------------------------------------------------
    def stamp(self, digest: bytes) -> bytes:
        """Submit ``digest`` to the calendars and return the ``.ots`` proof bytes.

        A random nonce is appended before submission so calendars learn nothing
        about the digest (as the reference client does).

        Raises:
            TimestampUnavailableError: If fewer than ``min_calendars`` calendars answer.
        """
        from opentimestamps.core.op import OpAppend, OpSHA256
        from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

        if len(digest) != 32:
            raise ValueError("digest must be 32 bytes (SHA-256)")
        detached = DetachedTimestampFile(OpSHA256(), Timestamp(digest))
        tip = detached.timestamp.ops.add(OpAppend(os.urandom(16))).ops.add(OpSHA256())
        answered: list[str] = []
        errors: list[str] = []
        for url in self.calendars:
            try:
                result = self._factory(url).submit(tip.msg, timeout=self.timeout)
            except Exception as exc:  # network, HTTP and parse errors alike
                errors.append(f"{url}: {exc}")
                continue
            tip.merge(result)
            answered.append(url)
        if len(answered) < self.min_calendars:
            raise TimestampUnavailableError(
                f"only {len(answered)} of {self.min_calendars} required calendars answered: "
                + "; ".join(errors)
            )
        return self._dump(detached)

    def upgrade(self, digest: bytes, proof: bytes) -> bytes:
        """Ask each pending calendar for a completed attestation; return the new proof."""
        from opentimestamps.core.notary import PendingAttestation

        detached = self._load(proof)
        if detached.file_digest != digest:
            raise ValueError("proof does not commit to this digest")
        for sub_stamp in list(_iter_timestamps(detached.timestamp)):
            for att in list(sub_stamp.attestations):
                if not isinstance(att, PendingAttestation):
                    continue
                try:
                    upgraded = self._factory(att.uri).get_timestamp(
                        sub_stamp.msg, timeout=self.timeout
                    )
                except Exception as exc:  # not yet anchored, or offline
                    log.info("calendar %s: %s", att.uri, exc)
                    continue
                sub_stamp.merge(upgraded)
        return self._dump(detached)

    def check(self, digest: bytes, proof: bytes) -> ProofCheck:
        """Parse ``proof`` and classify its attestations (no network)."""
        return check_ots_proof(digest, proof)


def _iter_timestamps(stamp: Any) -> Any:
    yield stamp
    for child in stamp.ops.values():
        yield from _iter_timestamps(child)


def check_ots_proof(digest: bytes, proof: bytes) -> ProofCheck:
    """Check that an ``.ots`` proof commits to ``digest`` and list its attestations.

    This is an offline, structural check. Confirming that a Bitcoin attestation's
    merkle root matches the real block header needs a Bitcoin node or block
    explorer; use ``ots verify`` for that.
    """
    if not ots_available():
        return ProofCheck(ProofStatus.UNAVAILABLE, "opentimestamps is not installed")
    from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
    from opentimestamps.core.op import OpSHA256

    try:
        detached = OtsTimestamper._load(proof)
    except Exception as exc:  # any deserialisation failure means a bad proof
        return ProofCheck(ProofStatus.INVALID, f"cannot parse proof: {exc}")
    if not isinstance(detached.file_hash_op, OpSHA256):
        return ProofCheck(ProofStatus.INVALID, "proof does not use SHA-256")
    if detached.file_digest != digest:
        return ProofCheck(ProofStatus.INVALID, "proof commits to a different hash")
    labels: list[str] = []
    anchored = False
    for _msg, att in detached.timestamp.all_attestations():
        if isinstance(att, BitcoinBlockHeaderAttestation):
            anchored = True
            labels.append(f"bitcoin:{att.height}")
        elif isinstance(att, PendingAttestation):
            labels.append(str(att.uri))
        else:
            labels.append(type(att).__name__)
    if anchored:
        return ProofCheck(ProofStatus.ANCHORED, "Bitcoin attestation present", tuple(labels))
    if labels:
        return ProofCheck(ProofStatus.PENDING, "awaiting Bitcoin confirmation", tuple(labels))
    return ProofCheck(ProofStatus.INVALID, "proof has no attestations")


def stamp_entries(
    entries: Sequence[LedgerEntry],
    timestamper: Timestamper | None,
    proofs_dir: Path = DEFAULT_PROOFS_DIR,
) -> dict[int, ProofStatus]:
    """Write a proof for every entry that lacks one; never raise for missing OTS.

    Returns the status of each entry's proof after the call, keyed by ``seq``.
    Entries whose stamping fails (no library, offline) are reported ``missing``
    and a warning is logged, so a weekly run degrades instead of failing.
    """
    results: dict[int, ProofStatus] = {}
    proofs_dir = Path(proofs_dir)
    for entry in entries:
        path = proof_path(entry, proofs_dir)
        digest = bytes.fromhex(entry.entry_hash)
        if path.exists():
            results[entry.seq] = (
                timestamper.check(digest, path.read_bytes()).status
                if timestamper is not None
                else check_ots_proof(digest, path.read_bytes()).status
            )
            continue
        if timestamper is None:
            log.warning("entry %d not timestamped: OpenTimestamps unavailable", entry.seq)
            results[entry.seq] = ProofStatus.MISSING
            continue
        try:
            proof = timestamper.stamp(digest)
        except TimestampUnavailableError as exc:
            log.warning("entry %d not timestamped: %s", entry.seq, exc)
            results[entry.seq] = ProofStatus.MISSING
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(proof)
        results[entry.seq] = timestamper.check(digest, proof).status
    return results


def default_timestamper(**kwargs: Any) -> OtsTimestamper | None:
    """Return an :class:`OtsTimestamper`, or ``None`` if OpenTimestamps is missing."""
    try:
        return OtsTimestamper(**kwargs)
    except TimestampUnavailableError as exc:
        log.warning("%s", exc)
        return None
