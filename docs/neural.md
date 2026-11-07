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
