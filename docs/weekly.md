# Weekly pipeline

```bash
uv run python -m kairos.commands.run weekly --cpu-only --report build/weekly-report.json
```

| Stage | Does | Entry points tried |
|---|---|---|
| `refresh` | Pulls `C(t)`, `H(t)`, events; writes the data manifest | `kairos.data.refresh:run_stage` |
| `fit` | Prophet null models, NeuralProphet ensembles | `kairos.models.weekly:run_stage` |
| `dials` | `δ`/`t_c`, `κ`, `h*`, `dG/dt` | `kairos.dials.weekly:run_stage` |
| `governor` | Jonas, Rawls, Collingridge; tripwire level | `kairos.governor.weekly:run_stage` |
| `ledger` | Hash, append, timestamp | `kairos.ledger.weekly:run_stage` |
| `export` | Static dashboard snapshot | `kairos.dashboard.export:run_stage` |

Each entry point takes a `RunContext` and may return a mapping, which later
stages read from `ctx.artifacts`. A stage whose module is not installed is
reported as `unavailable` and the run continues. Use `--strict` to make that an
error (exit code 2). A stage that raises stops the run (exit code 1) unless
`--keep-going` is given.

The GitHub Actions job `weekly.yml` runs this every Friday at 06:00 UTC on a
hosted runner with `--cpu-only` and commits any new files under `ledger/`.
