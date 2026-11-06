# Methodology

A short version of the method. The paper in `paper/` has the full argument.

## Null model

Prophet is fitted to `ln y`. A piecewise-linear trend on the log scale is
piecewise exponential growth: the "boring world". Persistent one-sided
residuals against it are evidence that the world is not boring.

## Acceleration: `δ` and `t_c`

Hyperbolic growth `dx/dt = k x^(1+δ)` reaches infinity at a finite time `t_c`.
With `g = d ln x / dt`,

$$ \ln g = \ln k + \delta \ln x, \qquad t_c - t = \frac{1}{\delta\, g}. $$

`g` is estimated from historical data with a cross-validated smoother, never
from the forecast segment (design note D1). Intervals come from a moving block
bootstrap (D4). LPPLS gives an independent `t_c`.

## Recursion: `κ`

The out-of-sample skill gained by adding capability's own lags to a
NeuralProphet model that already has the exogenous drivers. It is reported
against a permutation null (D9).

## Vinge Horizon: `h*`

The first forecast horizon at which the smoothed rolling-origin error, measured
in log units, crosses a factor tolerance (D2, D3). A shrinking `h*(t)` means
the world is getting harder to forecast.

## Jonas Gap: `dG/dt`

`G = ln C − ln H`, normalised at a base year. Only its velocity carries meaning
(D8).

## Decision rules

- **Jonas:** alarm when `P(danger) > 1/(1+λ)`. `λ` is published in YAML.
- **Rawls:** report the maximin decision next to the expected-value one.
- **Collingridge:** report lead time until the upper quantile crosses the
  threshold.
- **Popper:** every forecast is pre-registered in a hash-chained ledger.

| Level | Trigger |
|---|---|
| 0 Normal | All dials calm |
| 1 Watch | One dial in danger |
| 2 Warning | Two dials, or one on 3+ independent metrics |
| 3 Alarm | Three dials |
| 4 Kairos | All four dials |
