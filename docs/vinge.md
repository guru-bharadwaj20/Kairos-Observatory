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
