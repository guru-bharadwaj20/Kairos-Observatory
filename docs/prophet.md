# Prophet null models (phase 3)

Prophet on `ln y` is the null model of "normal" progress: piecewise-exponential growth with sparse changepoints. It stands for the boring world. The detectors below look for reality beating it.

| Module | Task | What it gives you |
|---|---|---|
| `kairos.models.logscale` | K-3.01 | `LogProphet`: fit on `ln y`, forecast in raw units. `exp(yhat)` is the **median**, and the interval bounds map exactly. |
| `kairos.models.null_model` | K-3.02 | `ProphetConfig`, `fit_null_model` → `NullModelFit` (rates in ln units per year) |
| `kairos.models.fit_all` | K-3.03 | `fit_all`: every series through `pmap`, several fits per task, wall time recorded |
| `kairos.models.changepoints` | K-3.04 | `changepoint_table`: dates, `delta`, rate before and after, magnitude, significance, rank |
| `kairos.models.acceleration` | K-3.05 | `acceleration_runs`: runs of consecutive significant slope increases |
| `kairos.models.blind` | K-3.06 | `blind_changepoint_check`: top-k accelerating shifts against 2012, 2017 and 2020–23, ±12 months |
| `kairos.models.logistic` | K-3.07 | `fit_capped_model`: `growth="logistic"` with `cap = ln H(t)/H(base) + ln headroom` |
| `kairos.models.cap_breach` | K-3.08 | `detect_cap_breaches`: episodes where observed `C` is above the cap |
| `kairos.models.residuals` | K-3.09 | `one_sided_residual_test`, `residuals_vs_fit`: sign test and upper CUSUM (k = 0.5, h = 5) |
| `kairos.models.sensitivity` | K-3.10 | `prior_sensitivity`: `changepoint_prior_scale` × `n_changepoints` grid and stable-region report |

## Configuration

`ProphetConfig` reads keyword arguments first, then `KAIROS_PROPHET_*` environment variables, then the `prophet:` section of `kairos.yaml`:

```yaml
prophet:
  changepoint_prior_scale: 0.3   # idea §9a
  n_changepoints: 25
  changepoint_range: 0.9         # Prophet's default is 0.8; 0.9 lets 2020-23 shifts in
  interval_width: 0.9
  horizon_months: 36
  yearly_seasonality: auto       # non-frontier series only
  frontier_patterns: ["*frontier*"]
```

Frontier series never get yearly seasonality.

## Units

- Rates and `delta` are in natural-log units per year. 0.69/yr means doubling every year.
- A changepoint counts as significant when `|delta| >= 0.01` ln/yr.
- The logistic model normalises `C` and `H` at a common base month (D8). Only the comparison between them means anything, never the level.

## Command line

```bash
kairos prophet fit   series.csv -o out/prophet     # summary.csv, forecast.csv, changepoints.csv, run.json
kairos prophet sweep series.csv -o out/prophet     # sensitivity.csv, sensitivity.md
```

Input is a `SeriesFrame` as CSV or Parquet.

## Validation status

The blind check is tested on `synthetic_ai_compute()`, a stylised series with shifts in 2012-06, 2017-06 and 2021-06 and noise σ = 0.15 on `ln y`. It passes for every seed tested (8, slow suite). It has not been run on real AI history yet, because that needs the phase-2 frontier series (K-2.06, K-2.21).
