"""Append-only, SHA-256 hash-chained JSONL ledger (D11).

Every line of ``ledger/forecasts.jsonl`` is the canonical JSON of one
:class:`LedgerEntry`. Each entry commits to

* its own ``payload`` (``payload_hash``),
* the entry before it (``prev_hash``; :data:`GENESIS_HASH` for the first entry),
* the data manifest it was produced from (``manifest_hash``, see K-2.04),

and ``entry_hash`` is the SHA-256 of all of the above. Changing, deleting or
reordering any line therefore breaks every hash after it.

Entries are never edited. A mistake is fixed by appending a ``correction`` entry
that names the hash of the entry it supersedes (Popper rule, CONTRIBUTING 2.5).
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kairos.ledger.canonical import canonical_json, is_sha256_hex, sha256_hex
from kairos.ledger.forecast import Forecast, forecast_hash

__all__ = [
    "DEFAULT_LEDGER_PATH",
    "DEFAULT_MANIFEST_PATH",
    "GENESIS_HASH",
    "EntryKind",
    "Ledger",
    "LedgerEntry",
    "LedgerError",
    "LedgerIntegrityError",
    "check_chain",
    "manifest_hash",
]

GENESIS_HASH: Final = "0" * 64
"""``prev_hash`` of the first entry."""

DEFAULT_LEDGER_PATH: Final = Path("ledger/forecasts.jsonl")
DEFAULT_MANIFEST_PATH: Final = Path("data/MANIFEST.json")

EntryKind = Literal["forecast", "intervention", "correction", "score"]


class LedgerError(RuntimeError):
    """Base class for ledger failures."""


class LedgerIntegrityError(LedgerError):
    """The ledger file does not match its own hash chain."""


class LedgerEntry(BaseModel):
    """One line of the ledger.

    Attributes:
        seq: 0-based position in the ledger.
        kind: ``forecast``, ``intervention`` (a logged policy event),
            ``correction`` (supersedes an earlier entry) or ``score``.
        recorded_at: UTC time the entry was appended.
        prev_hash: ``entry_hash`` of entry ``seq - 1``, or :data:`GENESIS_HASH`.
        manifest_hash: SHA-256 of the data manifest in force, if any.
        payload: Kind-specific body.
        payload_hash: SHA-256 of ``canonical_json(payload)``.
        entry_hash: SHA-256 of the canonical JSON of every other field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=0)
    kind: EntryKind
    recorded_at: datetime
    prev_hash: str
    manifest_hash: str | None = None
    payload: dict[str, Any]
    payload_hash: str
    entry_hash: str

    @field_validator("prev_hash", "payload_hash", "entry_hash")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not is_sha256_hex(value):
            raise ValueError(f"not a SHA-256 hex digest: {value!r}")
        return value

    @field_validator("manifest_hash")
    @classmethod
    def _hex_opt(cls, value: str | None) -> str | None:
        if value is not None and not is_sha256_hex(value):
            raise ValueError(f"not a SHA-256 hex digest: {value!r}")
        return value

    @field_validator("recorded_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")
        return value.astimezone(UTC)

    @staticmethod
    def compute_hash(
        *,
        seq: int,
        kind: str,
        recorded_at: datetime,
        prev_hash: str,
        manifest_hash: str | None,
        payload_hash: str,
    ) -> str:
        """Return the ``entry_hash`` for the given header fields."""
        header: dict[str, Any] = {
            "seq": seq,
            "kind": kind,
            "recorded_at": recorded_at,
            "prev_hash": prev_hash,
            "payload_hash": payload_hash,
        }
        if manifest_hash is not None:
            header["manifest_hash"] = manifest_hash
        return sha256_hex(header)

    @classmethod
    def create(
        cls,
        *,
        seq: int,
        kind: EntryKind,
        payload: dict[str, Any],
        prev_hash: str,
        manifest_hash: str | None = None,
        recorded_at: datetime | None = None,
    ) -> LedgerEntry:
        """Build a sealed entry, computing ``payload_hash`` and ``entry_hash``."""
        if recorded_at is not None and recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware")
        when = (recorded_at or datetime.now(UTC)).astimezone(UTC)
        # Round-trip the payload through canonical JSON so that what is stored is
        # exactly what was hashed (floats rounded, tuples as lists, dates as text).
        body: dict[str, Any] = json.loads(canonical_json(payload))
        p_hash = sha256_hex(body)
        e_hash = cls.compute_hash(
            seq=seq,
            kind=kind,
            recorded_at=when,
            prev_hash=prev_hash,
            manifest_hash=manifest_hash,
            payload_hash=p_hash,
        )
        return cls(
            seq=seq,
            kind=kind,
            recorded_at=when,
            prev_hash=prev_hash,
            manifest_hash=manifest_hash,
            payload=body,
            payload_hash=p_hash,
            entry_hash=e_hash,
        )

    def expected_payload_hash(self) -> str:
        """Recompute ``payload_hash`` from the payload."""
        return sha256_hex(self.payload)

    def expected_entry_hash(self) -> str:
        """Recompute ``entry_hash`` from the header."""
        return self.compute_hash(
            seq=self.seq,
            kind=self.kind,
            recorded_at=self.recorded_at,
            prev_hash=self.prev_hash,
            manifest_hash=self.manifest_hash,
            payload_hash=self.payload_hash,
        )

    def to_line(self) -> bytes:
        """Return the canonical JSONL line (with trailing newline)."""
        return canonical_json(self) + b"\n"

    def forecast(self) -> Forecast:
        """Return the forecast of a ``forecast`` entry."""
        if self.kind != "forecast":
            raise LedgerError(f"entry {self.seq} is a {self.kind}, not a forecast")
        return Forecast.model_validate(self.payload["forecast"])


def manifest_hash(path: Path = DEFAULT_MANIFEST_PATH) -> str:
    """Return the SHA-256 of the data manifest's canonical JSON.

    Hashing the parsed content (not the raw bytes) makes the hash independent of
    whitespace and key order in ``MANIFEST.json``.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    return sha256_hex(json.loads(Path(path).read_text(encoding="utf-8")))


def check_chain(entries: list[LedgerEntry]) -> None:
    """Raise :class:`LedgerIntegrityError` at the first broken link in ``entries``."""
    prev = GENESIS_HASH
    for i, entry in enumerate(entries):
        if entry.seq != i:
            raise LedgerIntegrityError(f"line {i + 1}: seq is {entry.seq}, expected {i}")
        if entry.prev_hash != prev:
            raise LedgerIntegrityError(f"entry {i}: prev_hash does not match entry {i - 1}")
        if entry.expected_payload_hash() != entry.payload_hash:
            raise LedgerIntegrityError(f"entry {i}: payload was modified (payload_hash mismatch)")
        if entry.expected_entry_hash() != entry.entry_hash:
            raise LedgerIntegrityError(f"entry {i}: header was modified (entry_hash mismatch)")
        prev = entry.entry_hash


def parse_line(line: str, lineno: int) -> LedgerEntry:
    """Parse one JSONL line into an entry.

    Raises:
        LedgerIntegrityError: If the line is not a valid entry.
    """
    try:
        return LedgerEntry.model_validate(json.loads(line))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LedgerIntegrityError(f"line {lineno}: not a valid ledger entry ({exc})") from exc


class Ledger:
    """A ledger file on disk.

    Args:
        path: The JSONL file; created (with parents) on first append.
    """

    def __init__(self, path: Path | str = DEFAULT_LEDGER_PATH) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"Ledger({str(self.path)!r})"

    def iter_entries(self) -> Iterator[LedgerEntry]:
        """Yield entries in file order, without checking the chain."""
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8", newline="\n") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.rstrip("\n")
                if not line.strip():
                    raise LedgerIntegrityError(f"line {lineno}: blank line in ledger")
                yield parse_line(line, lineno)

    def entries(self, *, check: bool = True) -> list[LedgerEntry]:
        """Return every entry, checking the chain first unless ``check`` is false."""
        result = list(self.iter_entries())
        if check:
            check_chain(result)
        return result

    def head(self) -> LedgerEntry | None:
        """Return the last entry (after checking the chain), or ``None`` if empty."""
        entries = self.entries()
        return entries[-1] if entries else None

    def forecasts(self) -> list[LedgerEntry]:
        """Return the ``forecast`` entries."""
        return [e for e in self.entries() if e.kind == "forecast"]

    def find(self, entry_hash: str) -> LedgerEntry:
        """Return the entry with ``entry_hash``.

        Raises:
            KeyError: If there is none.
        """
        for entry in self.entries():
            if entry.entry_hash == entry_hash:
                return entry
        raise KeyError(entry_hash)

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        lock = self.path.with_name(self.path.name + ".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise LedgerError(
                f"{lock} exists: another process is appending (delete it if stale)"
            ) from exc
        try:
            yield
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)

    def append(
        self,
        kind: EntryKind,
        payload: dict[str, Any],
        *,
        manifest_hash: str | None = None,
        recorded_at: datetime | None = None,
    ) -> LedgerEntry:
        """Seal ``payload`` as the next entry and append it to the file.

        The existing chain is checked first, so nothing is ever appended to a
        tampered ledger.

        Raises:
            LedgerIntegrityError: If the existing file fails :func:`check_chain`.
            LedgerError: If another append holds the lock.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            existing = self.entries()
            prev = existing[-1].entry_hash if existing else GENESIS_HASH
            if existing and recorded_at is not None and recorded_at < existing[-1].recorded_at:
                raise LedgerError("recorded_at must not go backwards")
            entry = LedgerEntry.create(
                seq=len(existing),
                kind=kind,
                payload=payload,
                prev_hash=prev,
                manifest_hash=manifest_hash,
                recorded_at=recorded_at,
            )
            with self.path.open("ab") as fh:
                fh.write(entry.to_line())
                fh.flush()
                os.fsync(fh.fileno())
        return entry

    def append_forecast(
        self,
        forecast: Forecast,
        *,
        manifest_hash: str,
        recorded_at: datetime | None = None,
    ) -> LedgerEntry:
        """Pre-register ``forecast`` against the data manifest ``manifest_hash``.

        Raises:
            LedgerError: If the same forecast (same hash) is already registered.
        """
        f_hash = forecast_hash(forecast)
        for entry in self.forecasts():
            if entry.payload.get("forecast_hash") == f_hash:
                raise LedgerError(f"forecast {f_hash[:12]} already registered as entry {entry.seq}")
        payload = {"forecast_hash": f_hash, "forecast": forecast.payload()}
        return self.append(
            "forecast", payload, manifest_hash=manifest_hash, recorded_at=recorded_at
        )

    def append_correction(
        self,
        supersedes: str,
        reason: str,
        *,
        replacement: dict[str, Any] | None = None,
        recorded_at: datetime | None = None,
    ) -> LedgerEntry:
        """Append a correction that supersedes the entry with hash ``supersedes``."""
        self.find(supersedes)
        payload: dict[str, Any] = {"supersedes": supersedes, "reason": reason}
        if replacement is not None:
            payload["replacement"] = replacement
        return self.append("correction", payload, recorded_at=recorded_at)
