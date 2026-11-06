# Security Policy

KAIROS publishes forecasts that people may act on, so the integrity of its code,
data and pre-registration ledger is a security matter, not only a bug.

## Supported versions

| Version | Supported |
|---|---|
| `main` | Yes |
| Tagged releases `< 0.1.0` | No (pre-release) |

## What to report privately

- Anything that could let someone alter, back-date or forge a ledger entry or its
  OpenTimestamps proof.
- Data-ingestion paths that could inject manipulated series (for example a
  compromised upstream source or a path traversal in a loader).
- Secrets or credentials committed by mistake.
- Vulnerabilities in the dashboard or CLI (code execution, SSRF, XSS).
- Supply-chain issues in pinned dependencies that affect KAIROS.

Ordinary modelling disagreements (a dial reading you think is wrong) are **not**
security issues; open a public issue so the discussion stays on the record.

## How to report

Use GitHub's **private vulnerability reporting**: *Security -> Report a
vulnerability* on this repository. Include steps to reproduce, the affected
commit and the impact you expect. Do not open a public issue for an unfixed
vulnerability.

## What to expect

- Acknowledgement within 7 days.
- An assessment and a fix plan within 30 days.
- Credit in the fix commit and release notes, unless you ask to stay anonymous.

Ledger corrections are always **appended**, never rewritten, including
corrections made because of a security report.
