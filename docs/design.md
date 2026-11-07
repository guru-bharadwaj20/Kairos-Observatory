# Design and engineering notes

The rules KAIROS holds itself to, the toolchain pins, how the work is split
between CPU and GPU on the reference workstation, and the design notes D1–D12
that came out of reviewing the original idea. Task ids such as `K-5.02` refer to
the build roadmap the project was delivered against.

## Non-negotiables (from the philosophy layer)

| Rule | Source | In practice |
|---|---|---|
| No self-certification | Löb | Code, data and forecasts are public. Don't accept data or funding that only a measured lab can verify. |
| Published parameters | Jonas | `λ` and every tripwire threshold live in versioned YAML, never as literals in code. |
| Falsifiability | Popper | Forecasts are hashed and committed before outcomes are known. Never edit a ledger entry; append corrections. |
| No lookahead | Popper | Backtests use only the data vintage available at each cutoff date. |
| Always show both sides | §13 | Any figure showing `C(t)` also shows `H(t)` and the recommended responses. |

## Development setup

### Python version: use 3.12, not the system 3.13

`neuralprophet 0.9.0` declares `requires_python <=3.12` and pins `numpy<2`. This workstation has Python 3.13, so the project pins **3.12** through `uv`:

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --extra dev --extra gpu
uv run kairos doctor
```

| Package | Constraint | Reason |
|---|---|---|
| Python | `==3.12.*` | neuralprophet upper bound |
| numpy | `>=1.26,<2` | neuralprophet pin; prophet and scipy accept 1.26 |
| prophet | `>=1.1.6` | cmdstanpy backend; ships its own CmdStan model |
| neuralprophet | `==0.9.*` | last release; brings pytorch-lightning |
| torch | CUDA 12.x wheel from the PyTorch index | Ada (sm_89) needs CUDA ≥ 11.8; check against the numpy 1.x pin |
| kaleido | `==0.2.1` (pinned by neuralprophet) | Known to hang on Windows; static image export goes through Plotly HTML instead |

### Everyday commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -m "not slow and not gpu"   # fast suite, runs in CI
uv run pytest -m gpu                       # this workstation only
uv run pytest -m slow -n 28                # backtests, parallel
```

## Hardware plan for this workstation

| Resource | Spec | What KAIROS uses it for |
|---|---|---|
| CPU | Intel Core i9-14900: 24 cores (8 P + 16 E), 32 threads | Prophet/Stan fits, rolling-origin backtests, block bootstraps, LPPLS float64 reference |
| GPU | NVIDIA RTX 4500 Ada, 24 GB, compute capability 8.9 | NeuralProphet seed ensembles and global models, vectorised bootstraps, batched LPPLS grid search |
| RAM | 128 GB | Hold every backtest fold and bootstrap draw in memory; no spilling needed for project-scale data |

### The CPU is the main workhorse

Prophet fits one Stan model per call on a single thread. The data is small (monthly series, about 200 points), so the work is **many small fits** that are embarrassingly parallel:

| Job | Fits per run (approx.) | Parallel strategy |
|---|:-:|---|
| Null models across all series | ~40 | `joblib` process pool |
| Vinge Horizon `h*(t)` (every monthly cutoff × every CV fold × every series) | ~50,000 | `prophet.diagnostics.cross_validation(parallel="processes")` inside an outer pool |
| `δ` / `t_c` block bootstrap | 10,000 per series | Process pool, chunked |
| Changepoint prior sensitivity sweep | ~400 | Process pool |

Settings, enforced by K-1.03 and K-1.04:

- **Default `n_jobs = 28`.** That leaves 4 threads for the OS, the GPU feeder and the dashboard.
- **Pin BLAS threads to 1 inside workers** (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) through `threadpoolctl`. Otherwise 28 workers × 32 BLAS threads oversubscribe the CPU and everything slows down.
- **Use dynamic scheduling** (`batch_size="auto"`, small chunks). P-cores and E-cores finish at different speeds, and static splits leave P-cores idle.
- Windows uses `spawn`, not `fork`. Worker entry points must be importable top-level functions, and the CmdStan model is compiled once and cached.

### The GPU is used where the work is wide, not deep

A single NeuralProphet fit on 200 monthly points is **faster on the CPU**, because kernel launch overhead dominates. The GPU pays off only when the work is batched:

| Job | Why it suits the GPU | Precision |
|---|---|---|
| LPPLS grid search: `t_c × m × ω` × windows × bootstraps (≈10⁸–10⁹ tiny 4-parameter least-squares solves) | Batched normal equations in one `torch` kernel | **float32** with centred, scaled inputs |
| `δ` bootstrap with 10⁵ resamples | Vectorised regression over a resample axis | float32 |
| NeuralProphet seed ensembles (≥ 20 seeds) for `κ` and global models over many series | Large batch, and the many fits amortise overhead | float32 (Lightning default) |

**FP64 warning:** Ada workstation GPUs run FP64 at about 1/64 of FP32 speed. Don't port float64 code to the GPU as-is. Every GPU kernel gets a CPU float64 reference implementation and an equivalence test (K-5.09). If float32 drifts, refine the best grid cells on the CPU in float64.

**Budget:** keep peak VRAM under 20 GB (chunk the grid) so the desktop and dashboard stay responsive. K-1.08 measures real throughput and writes it to `docs/benchmarks.md`. Later defaults (chunk sizes, CPU vs GPU per job) come from those numbers, not guesses.

### CI vs this workstation

| Where | Runs | Doesn't run |
|---|---|---|
| GitHub-hosted CI (Ubuntu + Windows) | Lint, types, fast tests, small synthetic backtests | `gpu`-marked tests, full backtests |
| This workstation (self-hosted runner, K-12.03) | Full weekly pipeline, GPU tests, `slow` backtests | — |

## Design notes from the idea review

Reading the idea document closely turned up the points below. Subtasks refer to them as **D1–D12**.

| # | Topic | Finding | Resolution | Task |
|:-:|---|---|---|:-:|
| D1 | `δ` from Prophet's trend (§9a) | Prophet's trend is piecewise linear in `ln x`, so `g = d ln x/dt` is a **step function**. The sketch also differentiates the 36-month *forecast* segment, where `g` is constant by construction. Both bias `δ` toward 0, so the detector is blind by design. | Estimate `g(t)` from **historical data only**, using a smoothing spline or local-linear derivative with bandwidth chosen by cross-validation. Cross-check with a direct nonlinear fit of `x = x0·(1 − t/t_c)^(−1/δ)`. | K-5.02, K-5.03 |
| D2 | Vinge Horizon metric (§9b) | MAPE is computed on `y = ln(FLOP)` (values around 57). A 25% error on a log value is a factor of about e^14 in real units. The tolerance means nothing. | Measure error in **log units directly**: absolute log error (factor error) and CRPS. Set the tolerance as a factor, e.g. "within 2×". | K-6.02 |
| D3 | Vinge Horizon definition | `ok["horizon"].max()` takes the longest horizon that passes. Errors aren't monotone in horizon, so one lucky long horizon inflates `h*`. | `h*` = **first** horizon where the smoothed error crosses tolerance. | K-6.03 |
| D4 | Standard errors of `δ` | `linregress` assumes independent errors. Here `ln g` and `ln x` come from the same autocorrelated series (errors in variables plus serial correlation), so `fit.stderr` is far too small. | Moving block bootstrap, refitting the smoother on every resample. | K-5.04, K-5.05 |
| D5 | Closed form for `T_remaining` | Checked: from `x = x0·(1 − t/t_c)^(−1/δ)`, `g = 1/(δ·(t_c − t))`, so `t_c − t = 1/(δ·g)`. Correct. | Keep it; add a delta-method interval and the bootstrap interval. | K-5.02 |
| D6 | Jonas threshold | Checked: raise the alarm when `λ·P > 1 − P`, i.e. `P > 1/(1+λ)`. Correct. | Implement as written. `λ` lives in public YAML. | K-8.02 |
| D7 | Frontier series construction | Epoch data are irregular, per-model points, not a monthly series. A running maximum produces flat steps that Prophet reads as many changepoints. | A dedicated **frontier envelope** builder (top-k or running max, plus a release-gap interpolation flag), with the method recorded in metadata. | K-2.06 |
| D8 | Level of the Jonas Gap | `ln C − ln H` subtracts quantities in different units, so the level of `G` is arbitrary and only `dG/dt` means anything. The idea's danger sign correctly uses `dG/dt`. | Normalise both indices at a base year. Present `dG/dt` as the headline and `G` only as relative to that base. | K-7.01 |
| D9 | Small data for NeuralProphet (§9c) | About 190 monthly points with `n_lags=12`, `n_forecasts=6`, 3 lagged regressors and events: easy to overfit, and `κ` is then noise. | Out-of-sample `κ` only, seed ensembles, early stopping, and a **permutation null** (shuffled self-lags) so `κ` is reported against chance. Cross-check with classical Granger/VAR tests. | K-4.08, K-4.10 |
| D10 | "Significant across several independent metrics" | Testing many metrics inflates false positives, and metrics from the same source aren't independent. | Benjamini–Hochberg across metrics, and metrics grouped by data source; "independent" means different source groups. | K-5.12, K-5.13 |
| D11 | Ledger immutability | Git history can be rewritten, so commit hashes alone don't prove when something was registered. | Hash-chained JSONL ledger, pushed publicly, plus **OpenTimestamps** anchoring. | K-10.02, K-10.04 |
| D12 | Toolchain | neuralprophet needs Python ≤ 3.12 and numpy < 2; the machine has 3.13. | Pin 3.12 with `uv` (see Development setup). | K-0.05 |

Sanity check on the hook: 13 November 2026 really is a Friday, so the von Foerster date in the idea is consistent. Validation task K-9.02 has to reproduce that `t_c` from the 1960 data.
