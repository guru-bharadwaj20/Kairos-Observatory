# KAIROS: The Self-Defeating Prophet

### An open early-warning system for runaway AI, built from mathematics, philosophy, Prophet and NeuralProphet

> *"The prophecy of doom is to be given greater heed than the prophecy of bliss."*
> Hans Jonas, *The Imperative of Responsibility* (1979)

---

## 0. Verdict: can the two ideas be mixed?

**Yes, and they fit together closely.**

- **A singularity is a time-series event.** In mathematics, a singularity is growth that runs faster than exponential and reaches infinity at a *finite* date `t_c`. Forecasting models are the natural instrument for catching that early.
- **Vernor Vinge's original 1993 definition of the "technological singularity" is about forecasting.** He described it as the point beyond which the future can no longer be predicted. A forecasting system can therefore measure its distance to the singularity by tracking how fast **its own forecasts are getting worse**.
- **Prophet and NeuralProphet** supply interpretable machinery: trend, changepoints, autoregression, lagged drivers, event effects and uncertainty. **Mathematics** supplies the singularity detectors. **Philosophy** decides what counts as danger and what to do about it. It also settles how to judge a prophet whose purpose is to be proven wrong.

So this is one project, not two.

---

## 1. A hook that lands next week

In 1960, Heinz von Foerster and colleagues published a paper in *Science* titled
**"Doomsday: Friday, 13 November, A.D. 2026."**
They fitted a hyperbolic curve to world population and found that it reached infinity on that date, **one week after this document was written (6 Nov 2026).**

Doomsday didn't happen. In the 1960s, population growth bent away from the hyperbola as the demographic transition set in. The singularity was real in the data and avoided in reality.

The lesson has two parts:

1. **Finite-time singularities can be seen in data decades in advance.**
2. **They can be bent.** The hard part is telling early whether a curve is bending.

KAIROS is built to answer that question for AI.

---

## 2. The real-world problem

| Problem | What it looks like today |
|---|---|
| **The Collingridge dilemma** (1980) | When a technology is young, it is easy to control, but we don't yet know its harms. Once the harms are clear, the technology is entrenched and hard to control. AI governance is stuck in this trap. |
| **Laws move in years, AI in months** | METR (2025) measured that the length of tasks AI agents can complete on their own has doubled roughly every 7 months since 2019. Epoch AI tracks frontier training compute growing roughly 4–5x per year. Legislation takes years. |
| **No neutral measuring instrument** | AI risk debates are mostly opinion against opinion. The Doomsday Clock is a committee's judgement, not a measurement. Lab safety frameworks are graded by the labs themselves. |
| **Nobody measures the other side** | We track what AI can do. Almost no one tracks systematically whether humanity's ability to *understand, test and govern* AI is keeping up. |

Weather has forecasting services, earthquakes have seismographs, and pandemics have surveillance networks. **Runaway AI has no instrument that gives falsifiable warnings tied to responses agreed in advance.**

---

## 3. The idea in one paragraph

**KAIROS** (Greek for *the decisive moment*, the right time to act, as opposed to *chronos*, clock time) is an open, independent observatory. It watches two curves, **what AI can do** and **what humans can oversee**, and keeps answering one question:

> **How much time is left while intervention is still cheap?**

Prophet and NeuralProphet serve as the null model of "normal" progress. Mathematical singularity detectors look for super-exponential acceleration. A philosophy layer turns ethical principles into explicit decision rules with published parameters. The outputs are **four measurable dials**, a **public clock** and **tripwire levels agreed in advance**. Every forecast is **cryptographically pre-registered**, so anyone can score it.

KAIROS is also designed as a **self-defeating prophecy**: it succeeds when its warnings trigger the action that makes them wrong.

---

## 4. Four core insights (the out-of-the-box part)

### Insight 1: The singularity is a mathematical object, so it can be detected

Exponential growth never reaches infinity:

```
dx/dt = k·x            ->   x(t) = x0·e^(k·t)
```

Super-exponential (hyperbolic) growth does, in finite time:

```
dx/dt = k·x^(1+δ),  δ > 0   ->   x(t) = x0·(1 − t/t_c)^(−1/δ)
```

**The tell:** under exponential growth, the growth *rate* stays constant however large the system gets. Approaching a singularity, the bigger it gets, the faster it grows. That is the signature of recursive self-improvement. To test for it, regress the log growth rate on the log level:

```
g(t) = d ln x / dt
ln g = ln k + δ · ln x
```

- `δ ≈ 0`: ordinary exponential progress
- `δ > 0`, sustained: finite-time singularity

The time remaining has a simple closed form:

```
T_remaining = 1 / (δ · g_now)
```

*Example:* if capability currently grows by one e-fold per year (`g = 1/yr`) and `δ = 0.1`, the curve blows up in about 10 years unless it bends.

A refinement is Didier Sornette's **LPPLS model** (Log-Periodic Power Law Singularity), which has been used to time financial bubbles:

```
ln x(t) = A + B·(t_c − t)^m + C·(t_c − t)^m · cos(ω·ln(t_c − t) − φ)
```

The log-periodic term picks up the *accelerating wobbles* that tend to come before a critical point.

### Insight 2: Vinge's singularity is where forecasting breaks, so measure how forecastable the world is

Turn Vinge's definition into a number, the **Vinge Horizon `h*(t)`**: the furthest ahead KAIROS can forecast AI capability within a fixed error tolerance, measured by rolling backtests. It works like Lorenz's limit on predictability in weather.

If `h*` shrinks from years to months to weeks, we are nearing the event horizon, whatever any single model says. **Here Prophet's failure is the signal.**

### Insight 3: Define the danger relationally, as capability against comprehension

An absolute capability threshold ("AGI = score X") is arbitrary. The real danger is a gap: AI capability `C(t)` growing faster than humanity's capacity `H(t)` to understand, evaluate and govern it.

```
Jonas Gap:        G(t) = ln C(t) − ln H(t)
Gap velocity:     dG/dt = (d ln C/dt) − (d ln H/dt)
```

This changes what *"stopping the singularity"* means. The goal is not to stop AI. It is to keep AI inside the envelope of human understanding. Under this definition the singularity can be prevented from **either side**: by slowing `C`, or by speeding up `H` through interpretability, evaluation science, education and faster institutions. That makes the project **pro-progress, not anti-AI**.

### Insight 4: A prophet that wants to be wrong

Robert Merton described the self-fulfilling prophecy, and it has a twin, the **self-defeating prophecy**. A warning system that works looks like a failure under naive scoring. This is the "preparedness paradox": Y2K "was nothing" *because* people fixed it.

KAIROS solves this with **conditional forecasts**:

> "If no new intervention occurs, metric X reaches Y by date Z (90% interval)."

Interventions are logged as **events** in NeuralProphet. Each forecast can then be scored against the counterfactual world, and the effect of each policy can be estimated. **KAIROS becomes an instrument for finding out which policies actually bend the curve.**

---

## 5. Where Prophet and NeuralProphet fit

| Job in KAIROS | Tool | Why this tool |
|---|---|---|
| **Null model of "normal" progress** | Prophet on `ln(y)` | A piecewise-linear trend on a log scale is piecewise exponential growth. The decomposition is interpretable, and the model tolerates gaps and outliers. |
| **Regime-shift detection** | Prophet changepoints | Finds past accelerations blind (2012 deep learning, 2017 transformers, 2020–23 LLM scaling). A run of slope *increases* is a candidate super-exponential signal. |
| **Human capacity ceiling** | Prophet `growth='logistic'` with a time-varying `cap` | Sets `cap = H(t)` to model capability as bounded by oversight capacity. Data breaking through the cap is itself a warning. |
| **Detecting recursive self-improvement** | NeuralProphet AR-Net (`n_lags`) plus lagged regressors | Measures how much of future capability growth comes from AI's own past capability *after* accounting for compute, money and people. That leftover coupling is the **recursion coefficient κ**. |
| **Policy and event effects** | NeuralProphet events and future regressors | Model releases, export controls, laws and incidents become events, and their estimated effects show which interventions bend the curve. |
| **Many metrics, many labs** | NeuralProphet global models | One model learns across related series (benchmarks, labs, countries) and shares strength where data is sparse. |
| **Uncertainty** | Prophet intervals, NeuralProphet `quantiles` | Interval width at each horizon feeds the Vinge Horizon. Jonas's rule acts on the *upper* quantile, not the median. |

> **Note:** Prophet isn't designed to extrapolate super-exponential growth, and that is deliberate. It stands for the *boring world*. When reality beats the forecast again and again, those one-sided residuals are evidence that the world is not boring.

---

## 6. The philosophy layer, written as mathematics

| Principle | Thinker | How KAIROS formalizes it |
|---|---|---|
| **Heuristic of fear / irreversibility** | Hans Jonas | Asymmetric loss: a missed catastrophe costs `λ` times a false alarm. Raise the alarm when `P(danger) > 1/(1+λ)`. **`λ` is the one number philosophy must supply.** It is published openly and set through public deliberation, not hidden in code. |
| **Veil of ignorance** | John Rawls | Decide by **maximin** across the scenario ensemble (do best in the worst plausible world), not by expected value alone. |
| **The control dilemma** | David Collingridge | Report **lead time** (months until the upper-quantile forecast crosses a threshold) next to a rising intervention-cost curve. The longer the lead time, the cheaper the fix. |
| **Falsifiability** | Karl Popper | Each weekly forecast is hashed (SHA-256) and committed to a public ledger *before* the outcome is known, then scored by CRPS and calibration. A prophet that can't be wrong is useless. |
| **Incompleteness / Löb's theorem** | Gödel, Löb | A formal system can't prove its own consistency. In practice: **no AI, and no lab, can be the final judge of its own safety.** KAIROS must be structurally independent, with open data, open code, independent replicas and no funding from the labs it measures. |
| **Goodhart's law** | Charles Goodhart | When a measure becomes a target, it stops being a good measure. KAIROS keeps rotating, held-out metrics. A gap between public benchmarks and held-out ones is a gaming signal in its own right. |
| **Self-defeating prophecy** | Merton, Jonas | Conditional forecasts with intervention events allow counterfactual scoring (Insight 4). |
| **Limits of prediction** | Vinge, Lorenz | The Vinge Horizon `h*(t)` (Insight 2). |

---

## 7. System architecture

```
   CAPABILITY  C(t)                        HUMAN CAPACITY  H(t)
   training compute, task-time horizon,    safety & interpretability researchers,
   benchmarks, AI share of AI research,    independent evaluations, law-making lag,
   cost per unit of capability             public AI literacy
                  \                                 /
                   \______________   ______________/
                                  \ /
                 EVENTS: model releases, laws, export controls, incidents
                                   |
                                   v
  +-------------------------------------------------------------------+
  |  FORECAST ENGINE                                                  |
  |    Prophet ............ null model of "normal" progress           |
  |    NeuralProphet ...... feedback loops, lagged drivers, events    |
  |    Singularity math ... hyperbolic delta-fit, LPPLS t_c           |
  |    Backtests .......... Vinge Horizon (how far can we still see?) |
  +-------------------------------------------------------------------+
                                   |
                                   v
  +-------------------------------------------------------------------+
  |  ETHICAL GOVERNOR  (philosophy written as mathematics)            |
  |    Jonas: asymmetric loss       Rawls: maximin over scenarios     |
  |    Löb: no self-certification   Popper: pre-registered forecasts  |
  |    Goodhart: rotating, held-out metrics                           |
  +-------------------------------------------------------------------+
                                   |
                                   v
        THE KAIROS CLOCK:  4 dials  ->  tripwire level  ->  public ledger
```

### Data sources to start with

**Capability `C(t)`:** Epoch AI datasets (training compute and parameters of notable models), METR task-time-horizon measurements, benchmark state-of-the-art histories, cost per unit of capability, and proxies for how much AI research is done *by AI*.

**Human capacity `H(t)`:** counts of AI safety and interpretability papers and researchers, the share of frontier models that receive independent pre-deployment evaluation, the lag between a new capability appearing and rules addressing it (OECD.AI policy database), safety funding, and public AI literacy surveys.

---

## 8. The four dials of the KAIROS Clock

| Dial | Symbol | What it measures | Danger sign |
|---|---|---|---|
| **Acceleration** | `δ`, `t_c` | Whether growth is super-exponential, and when it would blow up | `δ` significantly above 0 across several *independent* metrics, with the `t_c` interval narrowing |
| **Recursion** | `κ` | How much AI progress is driven by AI itself | `κ` rising across rolling windows |
| **Vinge Horizon** | `h*` | How far ahead we can still predict | `h*` shrinking |
| **Jonas Gap** | `G`, `dG/dt` | Capability outrunning comprehension | `dG/dt > 0`, sustained |

**κ, defined simply:** fit NeuralProphet with exogenous drivers only (compute, investment, talent), then again with capability's own lags added. `κ` is the relative gain in forecast skill from the self-lags, a Granger-style test of "AI improving AI".

### Tripwire levels (responses agreed in advance, like a fire code)

| Level | Trigger | Recommended response |
|---|---|---|
| **0: Normal** | All dials calm | Weekly publication |
| **1: Watch** | One dial in the danger zone | More data collection, public notice |
| **2: Warning** | Two dials, or one confirmed on 3 or more independent metrics | Mandatory third-party evaluation before new frontier deployments |
| **3: Alarm** | Three dials | Pause scaling past the current frontier until `H` catches up; emergency funding for the `H` side |
| **4: Kairos** | All four dials | The decisive moment, the last window in which intervention is still cheap |

The clock shows a number with error bars, for example **"Lead time: 31 months (90% CI: 18–52)"**, not a symbolic "minutes to midnight".

> **KAIROS recommends; humans decide.** It is an instrument, not an autonomous kill switch. Agreeing on responses in advance means nobody has to win a political argument in the middle of a crisis.

---

## 9. Code sketch (Python)

### a) Prophet as the null model, and the hyperbolicity test

```python
import numpy as np
import pandas as pd
from scipy import stats
from prophet import Prophet

# df: columns ds (monthly dates), y (e.g. frontier training compute in FLOP)
df["y"] = np.log(df["y"])                 # log scale: linear trend = exponential growth

m = Prophet(changepoint_prior_scale=0.3, interval_width=0.9)
m.fit(df)
fc = m.predict(m.make_future_dataframe(periods=36, freq="MS"))

# Smooth growth rate g = d ln x / dt from Prophet's trend (per year)
t_years = (fc["ds"] - fc["ds"].iloc[0]).dt.days / 365.25
ln_x = fc["trend"].values
g = np.gradient(ln_x, t_years)

# Hyperbolicity test: ln g = ln k + δ ln x
mask = g > 0
fit = stats.linregress(ln_x[mask], np.log(g[mask]))
delta = fit.slope                          # > 0  =>  finite-time singularity
T_remaining = 1 / (delta * g[mask][-1]) if delta > 0 else np.inf
print(f"δ = {delta:.3f} ± {fit.stderr:.3f},  time to singularity ≈ {T_remaining:.1f} yrs")
```

### b) The Vinge Horizon from Prophet's built-in backtesting

```python
from prophet.diagnostics import cross_validation, performance_metrics

cv = cross_validation(m, initial="1460 days", period="90 days", horizon="1095 days")
pm = performance_metrics(cv)
eps = 0.25                                  # tolerated error
ok = pm[pm["mape"] < eps]
vinge_horizon = ok["horizon"].max()         # recompute monthly; watch for shrinkage
```

### c) NeuralProphet for feedback loops and intervention events

```python
from neuralprophet import NeuralProphet

m2 = NeuralProphet(n_lags=12, n_forecasts=6, quantiles=[0.05, 0.95])
m2.add_lagged_regressor(["log_compute", "log_investment", "log_researchers"])
m2.add_events(["frontier_release", "major_regulation", "export_control"])
df_ev = m2.create_df_with_events(df_nn, events_df)
metrics = m2.fit(df_ev, freq="MS")
# κ = skill gain of this model vs. the same model with n_lags=0 (exogenous only)
```

---

## 10. Validation: prove it works before anyone trusts it

Backtest KAIROS on runaway curves whose endings we already know:

| Test case | What it checks |
|---|---|
| **World population** (von Foerster, 1960) | Would KAIROS have caught the bend (`δ` falling toward 0) in the 1960s–70s? |
| **Financial bubbles** (dot-com 2000, Bitcoin 2017) | The LPPLS detector against Sornette's published results |
| **COVID-19, early 2020** | Whether conditional forecasts and event modelling capture how lockdowns bent the curves |
| **Moore's law** | A long exponential with no singularity: a **false-alarm test** where KAIROS should stay at Level 0 |
| **AI history itself** | Does it find the 2012, 2017 and 2020–23 changepoints blind? |

**Scoring:** CRPS, calibration (do 90% intervals contain the truth about 90% of the time?), false-alarm rate, and warning lead time.

---

## 11. MVP roadmap (12 weeks, one person or a small team)

| Weeks | Milestone |
|---|---|
| 1–2 | Build the open dataset: `C(t)` and `H(t)` series plus an event calendar |
| 3–4 | Prophet null models on log scales; changepoint analysis of AI history |
| 5–6 | NeuralProphet models with lagged drivers and events; estimate `κ` |
| 7–8 | Hyperbolic `δ`-fit and LPPLS detector with bootstrap confidence intervals; historical backtests |
| 9–10 | Vinge Horizon and Jonas Gap; combine into the four dials and tripwire logic |
| 11–12 | Public dashboard, pre-registration ledger, write-up or paper |

**Tech stack:** Python, `prophet`, `neuralprophet`, `pandas`, `scipy`, `statsmodels`, and Streamlit or Plotly for the dashboard. A weekly GitHub Actions job refreshes data, issues forecasts and commits their hashes, which gives an immutable ledger.

**Deliverables:** an open dataset, an open-source `kairos` Python library, a live public dashboard, and a paper with a working title like *"Measuring the Distance to a Singularity: Forecastability as an Early-Warning Signal."*

---

## 12. Who uses it, and why it matters to people

- **AI safety institutes and regulators** get a neutral, data-driven trigger for evaluations instead of a political fight.
- **Legislators** get lead time ("about N months before this crosses X on current trends"), which is the way out of the Collingridge dilemma.
- **AI labs** get an external check. By Löb's logic, nobody can certify themselves.
- **Journalists and the public** get a calibrated number with error bars in place of hype and doom.
- **Funders** see from the Jonas Gap where money for human oversight capacity is most urgent.

---

## 13. Honest limitations, and how KAIROS handles them

| Limitation | Mitigation |
|---|---|
| Frontier data is sparse, delayed or secret | Global models, many proxies; growing secrecy is tracked as a risk signal in its own right |
| Labs may game metrics (Goodhart) | Rotating held-out metrics; divergence detection |
| "Capability" isn't one number | A metric ensemble; dials must agree across independent metrics |
| Singularity fits on noisy data raise false alarms | Bootstrap intervals, multi-dial confirmation, a published false-alarm rate |
| Prophet isn't built for super-exponential growth | Used on purpose as the *null* model, not the oracle |
| The clock could feed race dynamics ("we're close, sprint!") | Always publish the `H` side and the recommended responses alongside, and frame KAIROS as shared safety infrastructure |
| KAIROS can't stop anything by itself | True. It replaces argument with measurement and buys lead time. The decisions stay with people. |

---

## 14. Beyond AI

The same engine (null model, singularity detector, forecastability horizon, capability-against-capacity gap) applies to **any system that can run away**:

- synthetic-biology capability against biosecurity capacity
- financial bubbles against regulatory response
- pandemic growth against health-system capacity
- climate tipping points, with Scheffer et al.'s *critical slowing down* signals (rising autocorrelation and variance) added as a fifth dial

---

## 15. The pitch

> **"A seismograph for the intelligence explosion, built to be proven wrong."**

KAIROS doesn't try to stop AI. It measures whether humanity is keeping up and says, with error bars, how much time remains while the fix is still cheap. Every prediction it makes is public, timestamped and falsifiable. If it works, its alarms will look like false alarms, because people will have acted on them.

---

## References

- von Foerster, H., Mora, P. M., & Amiot, L. W. (1960). *Doomsday: Friday, 13 November, A.D. 2026.* Science, 132(3436), 1291–1295.
- Vinge, V. (1993). *The Coming Technological Singularity.* VISION-21 Symposium, NASA.
- Taylor, S. J., & Letham, B. (2018). *Forecasting at Scale.* The American Statistician, 72(1), 37–45. (Prophet)
- Triebe, O., et al. (2021). *NeuralProphet: Explainable Forecasting at Scale.* arXiv:2111.15397.
- Sornette, D. (2003). *Why Stock Markets Crash: Critical Events in Complex Financial Systems.* Princeton University Press. (LPPLS)
- Kremer, M. (1993). *Population Growth and Technological Change: One Million B.C. to 1990.* Quarterly Journal of Economics, 108(3).
- Collingridge, D. (1980). *The Social Control of Technology.*
- Jonas, H. (1979; English 1984). *The Imperative of Responsibility.*
- Rawls, J. (1971). *A Theory of Justice.*
- Popper, K. (1959). *The Logic of Scientific Discovery.*
- Merton, R. K. (1948). *The Self-Fulfilling Prophecy.* The Antioch Review.
- Lorenz, E. N. (1963). *Deterministic Nonperiodic Flow.* Journal of the Atmospheric Sciences.
- Scheffer, M., et al. (2009). *Early-warning signals for critical transitions.* Nature, 461, 53–59.
- Yudkowsky, E., & Herreshoff, M. (2013). *Tiling Agents for Self-Modifying AI, and the Löbian Obstacle.* MIRI.
- METR (2025). *Measuring AI Ability to Complete Long Tasks.* arXiv:2503.14499.
- Epoch AI. *Data on Notable AI Models.* epoch.ai
