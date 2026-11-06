# Vinge Horizon `h*`

`h*(t)` is the furthest horizon, in months, that KAIROS can forecast a series
at time `t` while staying inside a fixed error tolerance. Vinge (1993) defined
the singularity as the point past which the future can't be predicted, so a
shrinking `h*` is the signal. Code lives in `kairos.vinge`.

## 1. Input contract (for Phases 3, 4 and 5)

The Vinge Horizon never imports a concrete model. It needs one of two things.

**A `Forecaster`** (`kairos.vinge.schema.Forecaster`, a `typing.Protocol`):

```python
class Forecaster(Protocol):
    @property
    def interval_width(self) -> float: ...          # e.g. 0.90
    def fit_predict(self, history: SeriesFrame, horizon: int) -> ForecastFrame: ...
```

- `history` holds the rows of one series known at the cutoff.
- The result has exactly `horizon` monthly rows after the last `history.ds`,
  with columns `ds, yhat, yhat_lo, yhat_hi` in **raw positive units** (not
  logged). Extra quantiles go in optional `q_<p>` columns, e.g. `q_0.05`.
- The object must be picklable and deterministic, because fits run in worker
  processes and are cached by content.

**Or a ready `BacktestFrame`**: the forecast columns plus `series_id`,
`cutoff`, `horizon` (months, `>= 1`) and `y` (the realised value, raw units).
`from_prophet_cv` converts Prophet's `cross_validation` output.
`validate_backtest_frame` checks the schema.

## 2. Rolling-origin backtest (K-6.01)

`rolling_origin_backtest(frame, forecaster, horizon)` refits at every monthly
cutoff of every series. Each `(series, cutoff)` fit is one `pmap` task, cached
with `cached_fit`.

There is no lookahead. A row is visible at cutoff `c` only if `ds <= c` and,
when the frame has a `vintage` column (the publication date), `vintage <= c`
too. When a publication lag hides recent months, the model forecasts through
the gap, and only targets after `c` are scored. Actuals come from the latest
vintage.

`prophet_cross_validation(model, series_id, horizon)` wraps
`prophet.diagnostics.cross_validation(parallel="processes")` on explicit
monthly cutoffs. Prophet truncates by `ds` only, so use the generic harness
for series that get revised.

`LogDriftForecaster`, a random walk with drift on `ln y`, is the reference
baseline. It is also the forecaster the tests use.

## 3. Log-unit error metrics (K-6.02, D2)

MAPE on `ln y` means nothing here. `ln FLOP` is about 57, so a "25 %" error is
a factor of about e^14. `score_backtest` adds these columns, all on the log
scale:

| Column | Definition | Unit |
|---|---|---|
| `ale` | `abs(ln yhat - ln y)` | e-folds |
| `factor_error` | `exp(ale)`; 2.0 means "off by 2x" | factor |
| `crps` | CRPS of the forecast distribution of `ln y`. Uses the quantile (pinball) form when there are 3 or more `q_<p>` columns, else a Gaussian fitted to `yhat` and the band | e-folds |
| `band_efolds` | `ln yhat_hi - ln yhat_lo` | e-folds |
| `covered` | `yhat_lo <= y <= yhat_hi` | bool |

`error_curve(scored, metric)` aggregates by horizon, using the median by
default. Factor error is aggregated as `exp(median(ale))`.

## 4. The `h*` definition (K-6.03, D3)

The sketch's `ok["horizon"].max()` took the *longest* horizon that passed.
Errors aren't monotone, so one lucky long horizon inflated `h*`. KAIROS uses
the **first crossing** instead:

1. Build the error curve (median `ale` or `crps` per horizon), fill gaps, and
   smooth it with a centred moving average (`window=3` months).
2. `crossed_at` is the first horizon where the smoothed curve exceeds the
   tolerance. A tolerance factor `f` means `ln f` e-folds; `f = 2` is
   "within 2x".
3. `h* = crossed_at - 1`. If the curve never crosses, `h*` is the longest
   horizon tested and the result is `censored`, which makes it a lower bound.

`error_horizon(scored, tolerance_factor)` implements this rule.
`band_horizon(scored, k_efolds)` is the variant that applies the same rule to
the median width of the 90 % band. It needs no future actuals, so it can be
read at the live edge.
