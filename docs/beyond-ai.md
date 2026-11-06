# Beyond AI

The KAIROS engine (null model, singularity detector, forecastability horizon,
capability-against-capacity gap) fits any system that can run away. This page
shows two pieces that ship with the library.

## Pandemic growth against health-system capacity

`kairos.beyond.pandemic` treats hospital occupancy as `C(t)` and beds as
`H(t)`. The gap `G = ln C − ln H` reaches 0 on the day the system is
overwhelmed. The lead time uses the **upper** end of the recent growth
interval, the Jonas rule applied to an epidemic.

```python
from kairos.beyond.pandemic import capacity_gap, simulate_sir

free = simulate_sir(45)
gap = capacity_gap(free, capacity=50_000)
print(f"dG/dt = {gap.velocity:.3f}/day, lead time = {gap.lead_time_days:.0f} days")

locked = simulate_sir(120, lockdown_day=40)   # an intervention event bends the curve
```

COVID-19 in early 2020 is the historical test (K-9.05).

## Critical slowing down: an optional fifth dial

Near a tipping point, systems recover from small shocks more slowly (Scheffer
et al., 2009). `kairos.beyond.csd.csd_dial` detrends a series, computes rolling
lag-1 autocorrelation and variance, and is in `danger` when both trend upward
(Kendall's tau above the configured threshold).

```python
from kairos.beyond.csd import CSDConfig, csd_dial

dial = csd_dial(series, CSDConfig(window=0.5, bandwidth=0.1, tau_threshold=0.5))
print(dial.state, dial.tau_ar1, dial.tau_variance)
```

The fifth dial is **off by default**. It is not part of the AI tripwire levels;
it is offered for climate, ecological and financial applications where the
four AI dials have no direct analogue.

## API

::: kairos.beyond.pandemic

::: kairos.beyond.csd
