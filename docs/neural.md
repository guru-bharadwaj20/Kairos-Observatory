# NeuralProphet and the recursion dial κ

Phase 4 code lives in `src/kairos/neural/`. This page records the modelling
choices and the compute policy.

## The wrapper (`kairos.neural.model`)

- `NeuralConfig` is a frozen pydantic model with the hyper-parameters KAIROS
  varies: `n_lags`, `n_forecasts`, `epochs`, `batch_size`, `learning_rate`,
  `quantiles`, `n_changepoints`, `trend_reg`, `ar_reg`, `growth`,
  `early_stopping`. Seasonality is off: the series are monthly long-run trends.
- `trainer_config(device)` maps a `torch.device` from
  `kairos.compute.get_device()` onto the Lightning trainer (`accelerator`,
  `devices`) and turns off the progress bar, model summary, checkpoints and the
  `lightning_logs/` folder in the working directory (logs go to the temp dir).
- `fit_model` calls `seed_everything(seed)` and `neuralprophet.set_random_seed`
  before every fit, then fits with early stopping (NeuralProphet's patience of
  20 epochs) on the validation frame when one is given.
- `quiet_neuralprophet()` and `predict()` keep NeuralProphet and Lightning logs
  and warnings off the console.
- `synthetic_series()` simulates a capability series, its three drivers and a
  tunable self-coupling `phi`, so every test runs offline.

## Compute policy: CPU pool for small fits, GPU for wide work

`docs/benchmarks.md` (K-1.08) measured one 200-point NeuralProphet fit at
**0.64 s on the CPU and 1.21 s on the GPU**: kernel-launch and transfer
overhead dominate such a small model. Phase 4 therefore follows this rule:

| Job | Where | Why |
|---|---|---|
| One model, one series | CPU, in-process | Fastest single fit |
| Seed ensembles, paired `κ` models, permutation null, rolling windows | CPU pool via `pmap` (28 workers, one thread each) | Many independent small fits; about 28x the throughput of one GPU stream |
| Global model over many series (K-4.11) | `get_device()` (GPU when present) | One large model, wide batches |

Passing `device="cuda"` to the ensemble runner is supported; it then runs the
fits one after another in-process on the GPU, which is only worth it for large
global models.

## Lagged drivers (`kairos.neural.regressors`)

`design_frame()` joins the target and its drivers into one wide frame on the
months they share. Every driver is log-transformed and gets its own window:

| Column | Series | `n_lags` |
|---|---|--:|
| `log_compute` | `compute` | 6 |
| `log_investment` | `investment` | 12 |
| `log_researchers` | `researchers` | 12 |

Series IDs follow K-2.11; pass other `DriverSpec`s if Phase 2 names differ.

## Events (`kairos.neural.events`)

An `EventSpec` holds an event name, its months and an effect window
(`lower_window <= 0` months before, `upper_window >= 0` after).
`specs_from_calendar()` reads the K-2.17 calendar (`event`, `ds`, optional
window columns); `add_events()` registers the specs and `with_events()` runs
NeuralProphet's `create_df_with_events`. Effects are additive on the log
scale, so `exp(effect)` is the multiplicative change in raw units.

## Future regressors (`kairos.neural.future`)

Policies with a published schedule are `PolicySpec`s: a 0/1 step from the
effective month, an optional linear phase-in (`ramp_months`) and an optional
sunset (`until`). They enter as un-normalised future regressors, and
`future_frame()` fills in their scheduled values for the forecast months. A
policy that is constant over the training data (for example one that starts
after the data ends) is rejected: its effect cannot be learned.

## Quantiles and calibration (`kairos.neural.evaluation`)

Models forecast `quantiles=[0.05, 0.5, 0.95]` (`DEFAULT_QUANTILES`).
`holdout_split()` keeps the last `test_months` out of training;
`calibration_table()` reports, per horizon, the empirical share of held-out
values inside the 5–95 % band against the nominal 90 %, the shares below and
above, and the mean band width. `horizon_errors()` gives the out-of-sample MAE
per horizon in log units, the skill measure used for `κ`.

## Paired models (`kairos.neural.paired`)

`κ` compares two models that differ in one input channel only:

| Variant | Inputs |
|---|---|
| `exo` | trend + lagged drivers (+ events) |
| `self` | `exo` + the target's own last `self_lags` values |
| `null` | `self`, but the self channel is a block-shuffled surrogate of `y` |

The self channel is a lagged regressor holding a copy of `y`. With linear
layers this is the same function class as NeuralProphet's AR-Net (`n_lags`),
and it lets the permutation null use exactly the same architecture as `self`.
The surrogate permutes 12-month blocks of the first differences of `y` and
re-integrates them, so it keeps short-range autocorrelation and overall growth
but loses the alignment with the target. All variants share config, seed and
the hold-out window (`test_months`, default 24).

## Seed ensembles (`kairos.neural.ensemble`)

`run_seed_ensemble()` fits the paired models over `DEFAULT_N_SEEDS = 20`
seeds (child seeds from `derive_seed`, so an exo/self pair shares its seed).
Following the compute policy above, the roadmap's "batch on the GPU" is
replaced by the CPU pool: `ensemble_device(None | "auto")` is `"cpu"`, and
only an explicit `device="cuda"` runs the fits in-process on the GPU. Every
fit goes through `cached_fit`, and `summarize()` reports the mean, standard
deviation and 5/50/95 % quantiles of the out-of-sample MAE per variant.

## The recursion coefficient κ (`kairos.neural.kappa`)

With MAE the out-of-sample mean absolute error in log units, averaged over
horizons and seeds:

    κ = 1 − MAE_self / MAE_exo

- `κ = 0`: own lags add nothing once compute, money and people are known.
  `κ = 0.2`: they cut the forecast error by 20 %. `κ < 0`: they hurt.
- Interval: percentile bootstrap over the paired seeds (90 % by default).
- Null: `n_permutations` (default 49) refits with a block-shuffled self channel
  give `κ_null`; `p = (1 + #{κ_null ≥ κ}) / (1 + n_permutations)`, and
  `excess = κ − median(κ_null)` is the gain over chance.
- `kappa_reading()` returns the `"recursion"` `DialReading`: `value = κ`,
  `ci_lo/ci_hi` from the bootstrap, and `level = 1` only when `κ` is significant
  and rising across rolling windows (K-4.09); otherwise 0. `detail` carries
  the p-value, excess, MAEs, null and per-seed values.

The default `PairedSpec` (horizon 6, `self_lags = 12`, 24 hold-out months,
no changepoints, learning rate 0.1, batch 16, 300 epochs) was chosen on
synthetic series: with `phi = 0` it gives `κ ≈ 0`, with `phi = 0.95` a clearly
positive `κ`. Early stopping is off for `κ`: without a validation split it
watches the training loss and stopped high-learning-rate fits too early, and
overfitting is already handled by scoring out of sample against the null.

## Rolling κ(t) (`kairos.neural.rolling`)

`rolling_kappa()` re-estimates `κ` on trailing windows (default 120 months)
ending every `step` months (default = `test_months`, so hold-outs don't
overlap); all fits of all windows go to the pool as one batch. `trend_test()`
runs a one-sided Mann–Kendall test and a Theil–Sen slope with interval; `κ(t)`
is "rising" when `p < 0.05` and the slope is positive. `rolling_reading()`
gives the dial for the latest window with the trend and window table in
`detail`. Caveat: windows share data, so window estimates are positively
correlated and the trend p-value is optimistic; read it with the interval.

## Granger/VAR cross-check (`kairos.neural.granger`)

`self_lag_f_test()` is the classical twin of `κ`: a nested OLS F-test of the
target's own lags in an ARX regression on the lagged drivers (first
differences by default). `var_granger()` adds the VAR test that the drivers
Granger-cause the target (lag order by AIC). `agreement_table()` lists, per
series, `κ` and its permutation p-value next to the F-test and flags whether
both agree at `alpha = 0.05`.

## Global model (`kairos.neural.global_model`)

`global_frame()` stacks C-side series into NeuralProphet's `ds, y, ID`
layout. `fit_global()` trains one model on all of them (shared AR weights,
per-series trend via `trend_global_local="local"`, per-series normalisation)
on `get_device()`, the one GPU job in this phase. `compare_global_local()`
scores it against one model per series (CPU pool) on identical per-series
hold-outs and reports `gain = 1 − MAE_global / MAE_local`.

## Event effect table (`kairos.neural.effects`)

`event_effect_table()` fits drivers + events over a seed ensemble (CPU pool)
and reports, per event, the mean `event_<name>` component at the occurrence
months (log units), a 90 % seed interval, the peak inside the window, and
`factor = exp(effect)` as the multiplicative change in raw units.
`excludes_zero` flags intervals that don't cover 0. The interval measures
training noise only, so treat it as a screen, not proof that a policy works.

CLI: `kairos kappa synthetic --coupling 0.8` (offline demo) and
`kairos kappa estimate data.parquet --target <series_id>` print the reading
as JSON.
