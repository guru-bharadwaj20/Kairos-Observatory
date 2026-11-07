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

## External timestamping (K-10.04)

The chain proves order, not time, and git history can be rewritten. Each
entry's `entry_hash` is therefore anchored with
[OpenTimestamps](https://opentimestamps.org) (`kairos.ledger.timestamp`):

- The proof's file digest is the 32-byte `entry_hash` (SHA-256); a random
  nonce is appended before submission, as the reference client does.
- Proofs are standard `.ots` files at
  `ledger/proofs/<seq:06d>-<entry_hash[:16]>.ots`, committed with the ledger.
- At least 2 of the 4 default public calendars (the reference client's list)
  must answer, or the entry is left unstamped with a warning.
- The library is optional: `uv sync --extra ots`. Without it, or offline,
  stamping is skipped and proofs report `unavailable` / `missing`; nothing
  else breaks. Only the `opentimestamps` core package is imported
  (`otsclient`'s `python-bitcoinlib` fails to import on Windows).
- A fresh proof is `pending`. A few hours later, after the calendars' Bitcoin
  transaction confirms, `upgrade` turns it into `anchored`.

Offline checks are structural (the proof commits to the hash and lists its
attestations). For full Bitcoin verification use the reference client, e.g.
`ots verify -d <entry_hash> ledger/proofs/000000-....ots`.

## Scoring matured forecasts (K-10.05)

`score_matured(ledger, actuals, today=...)` scores every forecast whose horizon
end has passed and whose target dates all have observations, skipping
forecasts already scored or superseded by a correction. Each score is appended
as a `score` entry (`ForecastScore`): `forecast_entry`, `forecast_hash`,
`series_id`, `model`, `as_of`, `scored_on`, `coverage_level`, per-point
`points` (`ds`, `y`, `median`, `crps`, `pit`, `covered`, `log_gap`),
`mean_crps`, `coverage`, `counterfactual`, `interventions`, `mean_log_gap`
and resolved `claims` (`threshold`, `crossed`, `within_interval`).

CRPS is the quantile-loss approximation `2/K · Σ pinball`; PIT interpolates
the quantiles linearly. Interventions are logged with
`log_intervention(ledger, EventAssumption(...), series_ids=[...])`
(`intervention` entries: `{"event": {...}, "series_ids": [...]}`). If one
logged after a forecast takes effect within its horizon, targets its series
and was not already assumed, the score is `counterfactual` and
`mean_log_gap = mean log(y / median)` estimates the intervention's effect.

## Verifying (K-10.06)

```bash
kairos ledger verify                    # chain + offline proof checks; exit 1 on error
kairos ledger verify --require-proofs   # also fail on entries without a proof
kairos ledger verify --expect-head <entry_hash> --json
kairos ledger show                      # entries with proof status
```

`verify_ledger` reports every problem rather than stopping at the first:
invalid or non-canonical lines, CR line endings, `seq` gaps (deleted or
reordered entries), broken `prev_hash` links, payload/header/forecast hash
mismatches, corrupt or swapped proofs, and orphan proofs (a deleted tail
entry leaves its proof behind). `--expect-head` catches truncation of entries
that have no proof yet.

### Timestamping for real (manual, uses the network)

Tests never contact calendars. To anchor the public ledger:

```bash
uv sync --extra ots
kairos ledger stamp      # writes ledger/proofs/*.ots (pending)
# a few hours later
kairos ledger upgrade    # pending -> anchored
kairos ledger verify --require-proofs
git add ledger/ && git commit -m "data(ledger): ..."
```

## Public scorecard (K-10.07)

`kairos ledger scorecard [--out docs/scorecard.md]` rebuilds the table from
`score` entries alone (scores superseded by a correction are dropped). There is
one row per (series, model) plus `ALL`: `n_forecasts`, `n_points`,
`n_counterfactual`, `mean_crps` (series units; blank for `ALL`), empirical
`coverage` next to `nominal`, `pit_ks` (KS distance of PIT from uniform),
`claims_within` (share of conditional claims resolved inside their interval)
and `pit_hist` (10-bin PIT counts).
