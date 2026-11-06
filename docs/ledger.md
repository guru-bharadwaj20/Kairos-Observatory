# Pre-registration ledger

KAIROS commits every forecast to a public, append-only ledger *before* its
outcome is known (Popper rule, CONTRIBUTING 2.5; design note D11). Anyone can
recompute the chain and check that no entry was changed after the fact.

Code: `src/kairos/ledger/`. Data: `ledger/forecasts.jsonl`.

## Canonical JSON and hashing (K-10.01)

All hashes are SHA-256 over **canonical JSON** (`kairos.ledger.canonical`):

| Rule | Example |
|---|---|
| Keys sorted, no whitespace | `{"a":1,"b":2}` |
| ASCII only, `\uXXXX` escapes | `"δ"` for `δ` |
| Floats rounded to 15 significant digits, shortest round-trip text, always with `.` or exponent | `0.1 + 0.2` → `0.3`; `1e-7` → `1e-07`; `-0.0` → `0.0` |
| `NaN` / `±inf` rejected | — |
| Datetimes timezone-aware, written as UTC with `Z` | `"2026-11-06T12:00:00.000000Z"` |
| Dates ISO 8601 | `"2026-11-01"` |
| `None` fields of records omitted | adding an optional field later keeps old hashes |

A `Forecast` (`kairos.ledger.forecast`) holds `series_id`, `model`, `as_of`
(data vintage), `issued_at`, `unit`, quantile `points`, `code_version`,
`params` and optional `dials` (`DialReading`). `forecast_hash(f)` is the same
for the same forecast on every machine, and the same for the payload read back
from the ledger.

## Ledger file (K-10.02)

`ledger/forecasts.jsonl` holds one canonical JSON object per line (LF endings):

| Field | Type | Meaning |
|---|---|---|
| `seq` | int | 0-based position |
| `kind` | `forecast` \| `intervention` \| `correction` \| `score` | entry type |
| `recorded_at` | UTC timestamp | when the entry was appended |
| `prev_hash` | hex | `entry_hash` of the previous entry; 64 zeros for the first |
| `manifest_hash` | hex, optional | SHA-256 of `data/MANIFEST.json` (canonical JSON of its content) |
| `payload` | object | kind-specific body (below) |
| `payload_hash` | hex | `sha256(canonical(payload))` |
| `entry_hash` | hex | `sha256(canonical({seq, kind, recorded_at, prev_hash, manifest_hash, payload_hash}))` |

Payloads:

- `forecast`: `{"forecast_hash": ..., "forecast": <Forecast>}`; `manifest_hash` is required.
- `correction`: `{"supersedes": <entry_hash>, "reason": ..., "replacement"?: {...}}`.
  Entries are never edited; mistakes are corrected by appending.

`Ledger.append` checks the whole chain before writing, takes a
`forecasts.jsonl.lock` file so two processes cannot append at once, and
`fsync`s the line.

## Conditional forecasts (K-10.03)

A forecast can carry `events` (interventions already known at `as_of` and fed
to the model as event regressors; events dated after `as_of` are rejected) and
`claims` (`ConditionalClaim`), built with `Forecast.with_claims([Y, ...])`:

> If no new intervention, `frontier_compute` reaches 140 FLOP by 2027-02-01
> (90% interval: 2026-12-01 to 2027-04-01). Events assumed: eu_ai_act_gpai.

`by` is the first date the median reaches `Y`; the interval runs from the
first crossing of the 95% quantile to the first crossing of the 5% quantile
(`None` means "after the horizon"). Claim fields: `series_id`, `threshold`,
`unit`, `by`, `earliest`, `latest`, `horizon_end`, `coverage`, `condition`,
`assumed_events`.
