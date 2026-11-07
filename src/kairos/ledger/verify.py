"""Full ledger verification behind ``kairos ledger verify`` (K-10.06).

Unlike :func:`kairos.ledger.chain.check_chain`, which stops at the first broken
link, :func:`verify_ledger` reports *every* problem it finds:

* lines that are not valid entries;
* ``seq`` gaps or repeats (deleted or reordered lines);
* ``prev_hash`` links that do not match the previous line;
* ``payload_hash`` / ``entry_hash`` that do not match their content (edits);
* ``forecast_hash`` that does not match the stored forecast;
* proofs that are corrupt or commit to another hash;
* proof files with no matching entry (a deleted tail leaves orphan proofs);
* a head hash different from the one the caller expected (``--expect-head``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from kairos.ledger.chain import GENESIS_HASH, LedgerEntry, LedgerIntegrityError, parse_line
from kairos.ledger.forecast import forecast_hash
from kairos.ledger.timestamp import (
    DEFAULT_PROOFS_DIR,
    ProofStatus,
    Timestamper,
    check_ots_proof,
    proof_path,
)

__all__ = ["Issue", "VerifyReport", "verify_ledger"]

_PROOF_NAME = re.compile(r"^(\d{6})-([0-9a-f]{16})\.ots$")


@dataclass(frozen=True)
class Issue:
    """One verification finding.

    Attributes:
        severity: ``error`` fails verification; ``warning`` does not.
        message: Human-readable description.
        seq: Entry (or line index) concerned, if any.
    """

    severity: Literal["error", "warning"]
    message: str
    seq: int | None = None


@dataclass
class VerifyReport:
    """Result of :func:`verify_ledger`.

    Attributes:
        n_entries: Number of lines parsed as entries.
        head: ``entry_hash`` of the last entry, if any.
        proofs: Proof status per entry ``seq``.
        issues: All findings.
    """

    n_entries: int = 0
    head: str | None = None
    proofs: dict[int, ProofStatus] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether there are no errors."""
        return not self.errors

    @property
    def errors(self) -> list[Issue]:
        """The ``error`` issues."""
        return [i for i in self.issues if i.severity == "error"]

    def error(self, message: str, seq: int | None = None) -> None:
        """Record an error."""
        self.issues.append(Issue("error", message, seq))

    def warn(self, message: str, seq: int | None = None) -> None:
        """Record a warning."""
        self.issues.append(Issue("warning", message, seq))

    def raise_if_failed(self) -> None:
        """Raise :class:`LedgerIntegrityError` listing every error."""
        if self.errors:
            lines = "\n".join(f"  - {i.message}" for i in self.errors)
            raise LedgerIntegrityError(f"ledger verification failed:\n{lines}")


def _read_entries(path: Path, report: VerifyReport) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    if not path.exists():
        report.warn(f"{path} does not exist; nothing to verify")
        return entries
    data = path.read_bytes()
    if b"\r" in data:
        report.error("ledger contains CR characters (line endings must be LF)")
    if data and not data.endswith(b"\n"):
        report.error("ledger does not end with a newline (truncated write?)")
    for lineno, raw in enumerate(data.decode("utf-8", errors="replace").split("\n"), start=1):
        line = raw.rstrip("\r")
        if not line:
            continue
        try:
            entry = parse_line(line, lineno)
        except LedgerIntegrityError as exc:
            report.error(str(exc))
            continue
        if line.encode("utf-8") != entry.to_line().rstrip(b"\n"):
            report.error(f"line {lineno}: not in canonical form (re-serialised or edited)")
        entries.append(entry)
    return entries


def _check_entries(entries: list[LedgerEntry], report: VerifyReport) -> None:
    prev = GENESIS_HASH
    for i, entry in enumerate(entries):
        if entry.seq != i:
            report.error(
                f"position {i}: seq is {entry.seq} (entry deleted, inserted or reordered)", i
            )
        if entry.prev_hash != prev:
            report.error(f"entry {entry.seq}: prev_hash does not link to the previous entry", i)
        if entry.expected_payload_hash() != entry.payload_hash:
            report.error(f"entry {entry.seq}: payload_hash mismatch (payload edited)", i)
        if entry.expected_entry_hash() != entry.entry_hash:
            report.error(f"entry {entry.seq}: entry_hash mismatch (header edited)", i)
        if entry.kind == "forecast":
            if entry.manifest_hash is None:
                report.error(f"entry {entry.seq}: forecast without manifest_hash", i)
            try:
                actual = forecast_hash(entry.payload["forecast"])
            except (KeyError, ValidationError, ValueError, TypeError) as exc:
                report.error(f"entry {entry.seq}: forecast payload is invalid ({exc})", i)
            else:
                if actual != entry.payload.get("forecast_hash"):
                    report.error(f"entry {entry.seq}: forecast_hash mismatch", i)
        prev = entry.entry_hash


def _check_proofs(
    entries: list[LedgerEntry],
    proofs_dir: Path,
    report: VerifyReport,
    *,
    timestamper: Timestamper | None,
    require_proofs: bool,
) -> None:
    known = {proof_path(e, proofs_dir).name for e in entries}
    if proofs_dir.is_dir():
        for file in sorted(proofs_dir.glob("*.ots")):
            if file.name not in known:
                if _PROOF_NAME.match(file.name):
                    report.error(f"orphan proof {file.name}: its entry is missing from the ledger")
                else:
                    report.warn(f"unexpected file in proofs dir: {file.name}")
    for entry in entries:
        path = proof_path(entry, proofs_dir)
        if not path.exists():
            report.proofs[entry.seq] = ProofStatus.MISSING
            msg = f"entry {entry.seq}: no timestamp proof"
            report.error(msg, entry.seq) if require_proofs else report.warn(msg, entry.seq)
            continue
        digest = bytes.fromhex(entry.entry_hash)
        proof = path.read_bytes()
        check = timestamper.check(digest, proof) if timestamper else check_ots_proof(digest, proof)
        report.proofs[entry.seq] = check.status
        if check.status is ProofStatus.INVALID:
            report.error(f"entry {entry.seq}: invalid proof ({check.detail})", entry.seq)
        elif check.status is ProofStatus.UNAVAILABLE:
            report.warn(f"entry {entry.seq}: proof not checked ({check.detail})", entry.seq)


def verify_ledger(
    path: Path | str,
    proofs_dir: Path | str = DEFAULT_PROOFS_DIR,
    *,
    require_proofs: bool = False,
    expect_head: str | None = None,
    timestamper: Timestamper | None = None,
) -> VerifyReport:
    """Recompute the hash chain of ``path`` and check every proof in ``proofs_dir``.

    Args:
        path: The ledger JSONL file.
        proofs_dir: Directory holding the ``.ots`` proofs.
        require_proofs: Treat entries without a proof as errors (not warnings).
        expect_head: Known-good head ``entry_hash`` (e.g. from a published copy);
            catches deletion of trailing entries that have no proof yet.
        timestamper: Proof checker; defaults to the offline OpenTimestamps check.
    """
    report = VerifyReport()
    entries = _read_entries(Path(path), report)
    report.n_entries = len(entries)
    report.head = entries[-1].entry_hash if entries else None
    _check_entries(entries, report)
    _check_proofs(
        entries,
        Path(proofs_dir),
        report,
        timestamper=timestamper,
        require_proofs=require_proofs,
    )
    if expect_head is not None and expect_head not in {e.entry_hash for e in entries}:
        report.error(f"expected head {expect_head[:16]}... is not in the ledger")
    elif expect_head is not None and report.head != expect_head:
        report.warn("ledger has entries after the expected head")
    return report


def report_json(report: VerifyReport) -> str:
    """Serialise a report as JSON (for ``kairos ledger verify --json``)."""
    return json.dumps(
        {
            "ok": report.ok,
            "n_entries": report.n_entries,
            "head": report.head,
            "proofs": {str(k): v.value for k, v in report.proofs.items()},
            "issues": [
                {"severity": i.severity, "seq": i.seq, "message": i.message} for i in report.issues
            ],
        },
        indent=2,
    )
