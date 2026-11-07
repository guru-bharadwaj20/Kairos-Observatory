# Weekly pipeline

```bash
uv run python -m kairos.commands.run weekly --cpu-only --report build/weekly-report.json
```

| Stage | Does | Entry points tried |
|---|---|---|
| `refresh` | Pulls `C(t)`, `H(t)`, events; writes the data manifest | `kairos.weekly:refresh` |
| `fit` | Prophet null models on C/H series with 24+ positive months (NeuralProphet runs inside `κ`) | `kairos.weekly:fit` |
| `dials` | `δ`/`t_c`, `κ`, `h*`, `dG/dt` | `kairos.weekly:dials` |
| `governor` | Jonas, Rawls, Collingridge; tripwire level | `kairos.weekly:governor` |
| `ledger` | Hash, append, timestamp | `kairos.weekly:ledger` |
| `export` | Static dashboard snapshot | `kairos.weekly:export` |

Each entry point takes a `RunContext` and may return a mapping, which later
stages read from `ctx.artifacts`. A stage whose module is not installed is
reported as `unavailable` and the run continues. Use `--strict` to make that an
error (exit code 2). A stage that raises stops the run (exit code 1) unless
`--keep-going` is given.

The GitHub Actions job `weekly.yml` runs this every Friday at 06:00 UTC on a
hosted runner with `--cpu-only` and commits any new files under `ledger/`.

## Behaviour

- `refresh` downloads live data, uses the committed snapshots when
  `KAIROS_OFFLINE=1`, and falls back to them if a live download fails.
- `dials` reads each dial independently. A dial short of data (a `ValueError`
  from its package) is reported as `"<dial> skipped: <reason>"` in the stage
  detail and shown as unavailable in the snapshot; the run continues. With
  `--cpu-only`, `κ` uses a reduced ensemble (3 seeds, 3 null refits, 30 epochs).
  The Vinge backtest replays the current vintage pseudo-real-time.
- `governor` uses the normal-CI fallback for dials without draws.
- `ledger` appends the `idx.C` Prophet forecast to `ledger/forecasts.jsonl`
  (skipped if the same series/as-of/manifest is already registered) and tries
  to timestamp it unless `KAIROS_OFFLINE=1`.
- `export` writes `build/snapshot.json` (schema v1) and `build/site/`.
- The stage detail column of the run report summarises what each stage did.
