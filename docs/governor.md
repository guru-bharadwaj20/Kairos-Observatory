# Ethical governor and tripwires (Phase 8)

The governor turns the four dial readings (`kairos.types.DialReading`) into a
tripwire level 0–4. KAIROS recommends; humans decide.

## Public parameters

| File | Holds |
|---|---|
| `src/kairos/philosophy/policy.yaml` | Jonas `λ`, hysteresis band and dwell, per-dial danger rules, Rawls plausibility floor, Collingridge quantile/CI, intervention-cost curve, Goodhart test settings |
| `src/kairos/tripwires/tripwires.yaml` | The four counted dials, the confirmation rule (≥ 3 independent source groups) and levels 0–4 with their responses |

Both files are versioned and carry a changelog. Neither value appears as a literal in code.

## Reading contract

Dials send a `DialReading`. The governor reads these `detail` keys:

| Key | Type | Meaning |
|---|---|---|
| `samples` | `list[float]` | Bootstrap or ensemble draws of the dial's danger statistic (required unless `--allow-normal-fallback`) |
| `scenarios` | `dict[str, list[float]]` | Optional scenario ensemble; the decision uses the worst plausible one (Rawls) |
| `scenario_weights` | `dict[str, float]` | Optional weights (default equal) |
| `n_independent_groups` | `int` | Distinct source groups that agree (D10); ≥ 3 confirms a dial |
| `ci_level` | `float` | Interval level of `ci_lo`/`ci_hi` for the normal fallback (default 0.90) |

## Pipeline

1. `P(danger)` = share of draws on the danger side (`policy.yaml: danger`), with a Jeffreys interval.
2. With scenarios: decide on the worst plausible scenario and report the weighted mean beside it.
3. Feed `P(danger)` into a per-dial hysteresis machine whose upper edge is `1/(1+λ)`.
4. Level = `level_for(n_danger, n_confirmed)`: 0 none, 1 one, 2 two or one confirmed, 3 three, 4 all four.

## CLI

```bash
kairos governor spec
kairos governor evaluate readings.json [--history] [--spec PATH] [--policy PATH]
```

`evaluate` prints a `TripwireAlert` JSON with `level`, `name`, `trigger`,
`response`, `as_of`, `n_danger`, `n_confirmed`, the per-dial assessments,
`missing`/`ignored` dials, the spec and policy versions, `lam` and `threshold`.

## Other instruments

* `kairos.philosophy.collingridge.lead_time`: months until the upper-quantile path
  crosses a threshold, with a bootstrap interval, e.g. `31 months (90% CI: 18–52)`.
* `kairos.philosophy.intervention_cost.InterventionCostCurve`: relative cost
  `floor + (1 − floor)·exp(−L/τ)`.
* `kairos.tripwires.goodhart.detect_divergence`: public vs held-out log-trend gap
  (Newey-West SEs), reported as a `goodhart` reading that does not move the level.
