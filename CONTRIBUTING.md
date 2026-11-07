# Contributing to KAIROS Observatory

> *"A seismograph for the intelligence explosion, built to be proven wrong."*

This file is the build plan and the contribution guide. It breaks the project in [KAIROS_Project_Idea.md](KAIROS_Project_Idea.md) into **137 atomic subtasks** over 13 phases. Each subtask is sized to be **one commit**, so the finished project lands at about 137 commits, plus a few fix-ups, inside the 100–150 commit budget.

**Status legend:** ✅ done  ❌ not done

---

## Contents

1. [Progress overview](#1-progress-overview)
2. [Ground rules](#2-ground-rules)
3. [Development setup](#3-development-setup)
4. [Hardware plan for this workstation](#4-hardware-plan-for-this-workstation)
5. [Design notes from the idea review](#5-design-notes-from-the-idea-review)
6. [Subtasks by phase](#6-subtasks-by-phase)
7. [Definitions used in the tables](#7-definitions-used-in-the-tables)

---

## 1. Progress overview

| Phase | Name | Roadmap weeks | Subtasks | Done | Status |
|:-:|---|:-:|:-:|:-:|:-:|
| 0 | Foundation and governance | 0 | 14 | 14 | ✅ |
| 1 | Compute layer (CPU/GPU) | 0 | 8 | 8 | ✅ |
| 2 | Open dataset: `C(t)`, `H(t)`, events | 1–2 | 23 | 23 | ✅ |
| 3 | Prophet null models | 3–4 | 10 | 6 | ❌ |
| 4 | NeuralProphet and the recursion dial `κ` | 5–6 | 12 | 3 | ❌ |
| 5 | Singularity math: `δ`, `t_c`, LPPLS | 7–8 | 14 | 3 | ❌ |
| 6 | Vinge Horizon `h*` | 9 | 7 | 6 | ❌ |
| 7 | Jonas Gap `G`, `dG/dt` | 9–10 | 6 | 5 | ❌ |
| 8 | Ethical governor and tripwires | 10 | 9 | 7 | ❌ |
| 9 | Historical validation | 8, 10 | 9 | 2 | ❌ |
| 10 | Pre-registration ledger | 11 | 7 | 4 | ❌ |
| 11 | Public dashboard | 11–12 | 9 | 6 | ❌ |
| 12 | Automation, release, paper | 12 | 9 | 6 | ❌ |
| | **Total** | | **137** | **93** | **68%** |

A phase is ✅ only when every subtask in it is ✅.

---

## 2. Ground rules

### 2.1 One subtask, one commit

- Every commit maps to exactly one subtask ID, written in square brackets at the end of the subject line.
- The commit that finishes a subtask also flips its status from ❌ to ✅ in this file **and** updates the counts in [section 1](#1-progress-overview). Don't make separate "update status" commits; they waste the commit budget and split history.
- A fix-up for an already finished subtask reuses that subtask's ID: `fix(lppls): clamp m to filter bounds [K-5.08]`.
- Tests ship in the same commit as the code they test. There is no separate "add tests" phase.

### 2.2 Commit message format

[Conventional Commits](https://www.conventionalcommits.org/), with the task ID appended:

```
<type>(<scope>): <imperative summary> [K-<phase>.<nn>]

<optional body: why, not what>
```

| Type | Use for |
|---|---|
| `feat` | New capability (a dial, a loader, a detector) |
| `fix` | Bug fix to finished work |
| `docs` | Documentation only |
| `test` | Tests only (rare; prefer shipping tests with the code) |
| `perf` | GPU/CPU speedups with no change in behaviour |
| `refactor` | Restructuring with no change in behaviour |
| `chore` / `ci` / `build` | Tooling, workflows, packaging |
| `data` | Adding or refreshing a dataset |

Scopes match package modules: `data`, `prophet`, `np`, `singularity`, `lppls`, `vinge`, `jonas`, `governor`, `ledger`, `dashboard`, `compute`, `cli`.

### 2.3 Git identity

All commits in this repository are authored as **`guru-bharadwaj20`**. The repo-local git config is set to:

```bash
git config --local user.name  "guru-bharadwaj20"
git config --local user.email "184453467+guru-bharadwaj20@users.noreply.github.com"
```

Check `git config user.name` before committing on a new machine or clone.

### 2.4 Branches and reviews

- `main` must always pass CI.
- Work on `feat/K-<phase>.<nn>-short-name` branches, merged with fast-forward or rebase. Don't use merge commits; they add to the commit count without adding work.
- Every PR names its subtask ID and pastes the "Done when" text from the table as its checklist.

### 2.5 Non-negotiables (from the philosophy layer)

| Rule | Source | In practice |
|---|---|---|
| No self-certification | Löb | Code, data and forecasts are public. Don't accept data or funding that only a measured lab can verify. |
| Published parameters | Jonas | `λ` and every tripwire threshold live in versioned YAML, never as literals in code. |
| Falsifiability | Popper | Forecasts are hashed and committed before outcomes are known. Never edit a ledger entry; append corrections. |
| No lookahead | Popper | Backtests use only the data vintage available at each cutoff date. |
| Always show both sides | §13 | Any figure showing `C(t)` also shows `H(t)` and the recommended responses. |

---

## 3. Development setup

### 3.1 Python version: use 3.12, not the system 3.13

`neuralprophet 0.9.0` declares `requires_python <=3.12` and pins `numpy<2`. This workstation has Python 3.13, so the project pins **3.12** through `uv`:

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --extra dev --extra gpu
uv run kairos doctor          # available after K-1.01
```

| Package | Constraint | Reason |
|---|---|---|
| Python | `==3.12.*` | neuralprophet upper bound |
| numpy | `>=1.26,<2` | neuralprophet pin; prophet and scipy accept 1.26 |
| prophet | `>=1.1.6` | cmdstanpy backend; ships its own CmdStan model |
| neuralprophet | `==0.9.*` | last release; brings pytorch-lightning |
| torch | CUDA 12.x wheel from the PyTorch index | Ada (sm_89) needs CUDA ≥ 11.8; check against the numpy 1.x pin |
| kaleido | `==0.2.1` (pinned by neuralprophet) | Known to hang on Windows; static image export goes through Plotly HTML instead |

### 3.2 Everyday commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -m "not slow and not gpu"   # fast suite, runs in CI
uv run pytest -m gpu                       # this workstation only
uv run pytest -m slow -n 28                # backtests, parallel
```

---

## 4. Hardware plan for this workstation

| Resource | Spec | What KAIROS uses it for |
|---|---|---|
| CPU | Intel Core i9-14900: 24 cores (8 P + 16 E), 32 threads | Prophet/Stan fits, rolling-origin backtests, block bootstraps, LPPLS float64 reference |
| GPU | NVIDIA RTX 4500 Ada, 24 GB, compute capability 8.9 | NeuralProphet seed ensembles and global models, vectorised bootstraps, batched LPPLS grid search |
| RAM | 128 GB | Hold every backtest fold and bootstrap draw in memory; no spilling needed for project-scale data |

### 4.1 The CPU is the main workhorse

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

### 4.2 The GPU is used where the work is wide, not deep

A single NeuralProphet fit on 200 monthly points is **faster on the CPU**, because kernel launch overhead dominates. The GPU pays off only when the work is batched:

| Job | Why it suits the GPU | Precision |
|---|---|---|
| LPPLS grid search: `t_c × m × ω` × windows × bootstraps (≈10⁸–10⁹ tiny 4-parameter least-squares solves) | Batched normal equations in one `torch` kernel | **float32** with centred, scaled inputs |
| `δ` bootstrap with 10⁵ resamples | Vectorised regression over a resample axis | float32 |
| NeuralProphet seed ensembles (≥ 20 seeds) for `κ` and global models over many series | Large batch, and the many fits amortise overhead | float32 (Lightning default) |

**FP64 warning:** Ada workstation GPUs run FP64 at about 1/64 of FP32 speed. Don't port float64 code to the GPU as-is. Every GPU kernel gets a CPU float64 reference implementation and an equivalence test (K-5.09). If float32 drifts, refine the best grid cells on the CPU in float64.

**Budget:** keep peak VRAM under 20 GB (chunk the grid) so the desktop and dashboard stay responsive. K-1.08 measures real throughput and writes it to `docs/benchmarks.md`. Later defaults (chunk sizes, CPU vs GPU per job) come from those numbers, not guesses.

### 4.3 CI vs this workstation

| Where | Runs | Doesn't run |
|---|---|---|
| GitHub-hosted CI (Ubuntu + Windows) | Lint, types, fast tests, small synthetic backtests | `gpu`-marked tests, full backtests |
| This workstation (self-hosted runner, K-12.03) | Full weekly pipeline, GPU tests, `slow` backtests | — |

---

## 5. Design notes from the idea review

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
| D12 | Toolchain | neuralprophet needs Python ≤ 3.12 and numpy < 2; the machine has 3.13. | Pin 3.12 with `uv` (section 3.1). | K-0.05 |

Sanity check on the hook: 13 November 2026 really is a Friday, so the von Foerster date in the idea is consistent. Validation task K-9.02 has to reproduce that `t_c` from the 1960 data.

---

## 6. Subtasks by phase

**Columns:** **ID** is the commit tag. **Done when** is the acceptance criterion. **HW** is where the work runs: `CPU` is a single process, `CPU×N` is the parallel pool, `GPU` is CUDA, `—` means no compute. **Deps** lists the tasks that must be ✅ first.

### Phase 0: Foundation and governance

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-0.01 | Add project idea document | `KAIROS_Project_Idea.md` committed | — | — | ✅ |
| K-0.02 | Add licence | Apache-2.0 `LICENSE` + `NOTICE` committed | — | — | ✅ |
| K-0.03 | Add contribution guide and roadmap | This file committed with all 137 subtasks | — | — | ✅ |
| K-0.04 | README | Pitch, the four dials, tripwire table, quickstart, hardware notes, links to the idea doc and this file | — | 0.03 | ✅ |
| K-0.05 | `pyproject.toml` with `uv` | Python `==3.12.*`, `src/kairos/` layout, extras `dev`, `gpu`, `dashboard`; `uv sync` succeeds (D12) | CPU | — | ✅ |
| K-0.06 | Lock file and CUDA wheel index | `uv.lock` committed; `torch.cuda.is_available()` is `True` on this machine | GPU | 0.05 | ✅ |
| K-0.07 | `.gitignore` and `.gitattributes` | Ignores venv, `data/raw/`, `lightning_logs/`, CmdStan build output, caches; LF line endings enforced | — | — | ✅ |
| K-0.08 | Lint and type tooling | `ruff`, `ruff format`, `mypy --strict` on `src/`, `pre-commit` hooks installed and passing | CPU | 0.05 | ✅ |
| K-0.09 | Test scaffold | `pytest`, `pytest-xdist`, `hypothesis`, coverage; markers `slow` and `gpu` registered; one smoke test | CPU | 0.05 | ✅ |
| K-0.10 | CI workflow | GitHub Actions on Ubuntu + Windows: lint, types, fast tests; GPU tests skipped | CPU | 0.08, 0.09 | ✅ |
| K-0.11 | Community files | `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR templates that require a task ID | — | — | ✅ |
| K-0.12 | Governance and independence policy | `GOVERNANCE.md`: no funding from measured labs (Löb), conflict-of-interest disclosure, how `λ` is set through public deliberation | — | — | ✅ |
| K-0.13 | Data licence | `LICENSE-DATA` (CC BY 4.0) for KAIROS-built datasets and published forecasts; third-party licences listed | — | — | ✅ |
| K-0.14 | Citation metadata | `CITATION.cff` validated with `cffconvert` | — | — | ✅ |

### Phase 1: Compute layer (CPU/GPU)

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-1.01 | `kairos doctor` command | Reports P/E core split, RAM, CUDA, GPU name/VRAM, torch build, CmdStan status; exits non-zero if something is missing | CPU, GPU | 0.06 | ✅ |
| K-1.02 | Settings and logging | `pydantic-settings` + YAML for `n_jobs`, device, seeds, paths; structured logs with a run ID on every line | — | 0.05 | ✅ |
| K-1.03 | Thread governance | Workers start with BLAS/OMP threads = 1 through `threadpoolctl`; a test proves no oversubscription | CPU×N | 1.02 | ✅ |
| K-1.04 | Parallel executor | `kairos.compute.pmap()` on joblib loky; default 28 workers, auto batching for hybrid cores, Windows-`spawn` safe | CPU×N | 1.03 | ✅ |
| K-1.05 | Device selector | `get_device()` returns CUDA or falls back to CPU; float32 policy for GPU, float64 for CPU references | GPU | 1.02 | ✅ |
| K-1.06 | Reproducibility | One `seed_everything()` covering numpy, torch, CUDA, CmdStan and Lightning; the same seed reproduces outputs bit for bit on CPU | CPU, GPU | 1.05 | ✅ |
| K-1.07 | Fit cache | `joblib.Memory` keyed by data hash + params + code version; reruns skip finished fits | — | 1.02 | ✅ |
| K-1.08 | Hardware benchmark | Script measures Prophet fits/s (1 vs 28 workers), NeuralProphet CPU vs GPU, batched least squares on GPU vs CPU; results in `docs/benchmarks.md` | CPU×N, GPU | 1.04, 1.05 | ✅ |

### Phase 2: Open dataset — `C(t)`, `H(t)`, events

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-2.01 | Series schema | `pandera` schema: `ds`, `y`, `series_id`, `side` (C/H/driver/validation), `unit`, `source`, `source_group`, `vintage`, `y_lo`, `y_hi` | — | 0.05 | ✅ |
| K-2.02 | Event schema | `event`, `ds`, `type` (release, regulation, export control, incident), `lower_window`, `upper_window`, `source_url` | — | 2.01 | ✅ |
| K-2.03 | Storage and catalog | `data/raw/` → `data/processed/` Parquet; DuckDB catalog view across all series | CPU | 2.01 | ✅ |
| K-2.04 | Data manifest | SHA-256 of every processed file in `data/MANIFEST.json`, regenerated on refresh (feeds the ledger) | — | 2.03 | ✅ |
| K-2.05 | Epoch AI notable models loader | Training compute, parameters, date, organisation; licence recorded | CPU | 2.03 | ✅ |
| K-2.06 | Frontier envelope builder | Irregular releases → monthly frontier (top-k or running max), gap flags, method in metadata (D7) | CPU | 2.05 | ✅ |
| K-2.07 | METR task-horizon loader | Sparse points with their confidence intervals kept in `y_lo`/`y_hi` | CPU | 2.03 | ✅ |
| K-2.08 | Benchmark SOTA histories | SOTA per benchmark over time; logit transform for bounded scores so saturation doesn't look like a slowdown | CPU | 2.03 | ✅ |
| K-2.09 | Cost per unit of capability | Price per million tokens at a fixed benchmark score, over time | CPU | 2.08 | ✅ |
| K-2.10 | AI share of AI research proxy | Documented proxy series with method and caveats written down | CPU | 2.03 | ✅ |
| K-2.11 | Exogenous drivers | `log_compute`, `log_investment`, `log_researchers` monthly series for `κ` | CPU | 2.05 | ✅ |
| K-2.12 | H: safety and interpretability publications | Monthly counts from the OpenAlex/arXiv APIs with a fixed, versioned query | CPU | 2.03 | ✅ |
| K-2.13 | H: researchers and safety funding | Headcount and funding series with uncertainty bands | CPU | 2.03 | ✅ |
| K-2.14 | H: independent pre-deployment evaluations | Share of frontier releases with a third-party evaluation, per quarter | CPU | 2.06 | ✅ |
| K-2.15 | H: regulatory lag | Lag from a capability first appearing to the first rule addressing it, from the OECD.AI policy database | CPU | 2.03 | ✅ |
| K-2.16 | H: public AI literacy | Survey points, interpolated with widening uncertainty between surveys | CPU | 2.03 | ✅ |
| K-2.17 | Event calendar v1 | Frontier releases 2012–2026, major laws, export controls, notable incidents; ≥ 1 source URL each | — | 2.02 | ✅ |
| K-2.18 | Validation data: population and Moore's law | World population (Kremer/UN historical), transistor counts | CPU | 2.03 | ✅ |
| K-2.19 | Validation data: bubbles and COVID | Nasdaq 1995–2001, Bitcoin 2015–2018, OWID COVID cases with lockdown events | CPU | 2.03 | ✅ |
| K-2.20 | Data quality checks | Date monotonicity, units, duplicates, gaps, outlier flags; run on every refresh | CPU | 2.03 | ✅ |
| K-2.21 | Composite indices `C(t)` and `H(t)` | Normalised growth rates combined by geometric mean (PCA as a variant); leave-one-out sensitivity table | CPU | 2.06–2.16 | ✅ |
| K-2.22 | `kairos data refresh` | One command fetches, validates, writes Parquet and updates the manifest | CPU | 2.04, 2.20 | ✅ |
| K-2.23 | Datasheet | `docs/data.md`: provenance, licence, known biases and secrecy notes for each series | — | 2.21 | ✅ |

### Phase 3: Prophet null models

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-3.01 | Log-scale wrapper | Fit on `ln y`, back-transform the median and intervals; round-trip test | CPU | 0.05 | ✅ |
| K-3.02 | Null model per series | Defaults: `interval_width=0.9`, no yearly seasonality on frontier series, `changepoint_prior_scale` from config | CPU | 3.01, 2.21 | ✅ |
| K-3.03 | Parallel fit of all series | Every series fitted through `pmap`; wall time recorded | CPU×N | 3.02, 1.04 | ✅ |
| K-3.04 | Changepoint extraction | Table of changepoint dates, slope deltas and magnitudes per series | CPU | 3.02 | ✅ |
| K-3.05 | Acceleration-run detector | Flags runs of consecutive positive slope deltas as candidate super-exponential segments | CPU | 3.04 | ✅ |
| K-3.06 | Blind changepoints on AI history | Finds 2012, 2017 and 2020–23 within ±12 months without being told | CPU | 3.04 | ✅ |
| K-3.07 | Logistic model with `cap = H(t)` | `growth="logistic"` with a time-varying cap from the H index | CPU | 3.02, 2.21 | ❌ |
| K-3.08 | Cap-breach detector | Reports when observed `C` breaks through the H-bounded cap | CPU | 3.07 | ❌ |
| K-3.09 | One-sided residual test | Sign test and CUSUM on "reality beats the forecast" residuals | CPU | 3.02 | ❌ |
| K-3.10 | Prior sensitivity sweep | Grid over `changepoint_prior_scale` × `n_changepoints`; stable-region report | CPU×N | 3.03 | ❌ |

### Phase 4: NeuralProphet and the recursion dial `κ`

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-4.01 | NeuralProphet wrapper | `trainer_config` picks GPU or CPU through `get_device()`; early stopping; quiet logs | CPU, GPU | 1.05 | ✅ |
| K-4.02 | Lagged regressors | `log_compute`, `log_investment`, `log_researchers` added with their own lag windows | CPU | 4.01, 2.11 | ✅ |
| K-4.03 | Events | Event calendar wired in through `create_df_with_events` with windows | CPU | 4.01, 2.17 | ✅ |
| K-4.04 | Future regressors | Policies known in advance (scheduled laws) entered as future regressors | CPU | 4.03 | ❌ |
| K-4.05 | Quantiles and calibration | `quantiles=[0.05, 0.5, 0.95]`; empirical coverage reported on held-out data | CPU | 4.01 | ❌ |
| K-4.06 | Paired models for `κ` | Exogenous-only model and the same model with self-lags (`n_lags`), identical splits | CPU, GPU | 4.02 | ❌ |
| K-4.07 | Seed ensemble runner | ≥ 20 seeds per model in a batch on the GPU; mean and spread of skill | GPU | 4.06, 1.06 | ❌ |
| K-4.08 | `κ` estimator | `κ` = relative out-of-sample skill gain from self-lags, with a permutation null (D9) | GPU | 4.07 | ❌ |
| K-4.09 | Rolling `κ(t)` | `κ` recomputed over rolling windows; trend test for "rising" | GPU | 4.08 | ❌ |
| K-4.10 | Granger/VAR cross-check | `statsmodels` Granger test on the same series; agreement table with `κ` | CPU | 4.08 | ❌ |
| K-4.11 | Global model | One NeuralProphet model across all C-side series and labs; compared with per-series models | GPU | 4.07 | ❌ |
| K-4.12 | Event effect table | Estimated effect and interval for each intervention: "which policies bend the curve" | CPU | 4.03 | ❌ |

### Phase 5: Singularity math — `δ`, `t_c`, LPPLS

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-5.01 | Synthetic test series | Generators for exponential (`δ = 0`), hyperbolic (known `δ`, `t_c`) and logistic curves, with noise | CPU | 0.09 | ✅ |
| K-5.02 | Growth-rate estimator and `δ` regression | `g(t)` from smoothing-spline derivative on history only (D1); `ln g = ln k + δ ln x`; `T_remaining = 1/(δ·g)` with delta-method interval (D5) | CPU | 5.01 | ✅ |
| K-5.03 | Direct hyperbolic fit | `scipy.optimize.least_squares` on `x0·(1 − t/t_c)^(−1/δ)`; agrees with K-5.02 on synthetic data | CPU | 5.01 | ✅ |
| K-5.04 | Block bootstrap (CPU) | Moving block bootstrap for `δ` and `t_c` with smoother refit per draw (D4) | CPU×N | 5.02, 1.04 | ❌ |
| K-5.05 | Vectorised bootstrap (GPU) | 10⁵ resamples in batched `torch`; matches K-5.04 distribution (KS test) | GPU | 5.04, 1.05 | ❌ |
| K-5.06 | Coverage check | 90% bootstrap intervals cover the true synthetic `δ` in 85–95% of trials | CPU×N | 5.04 | ❌ |
| K-5.07 | LPPLS reference (CPU float64) | Filimonov–Sornette form: linear `A, B, C1, C2` by OLS, nonlinear `t_c, m, ω` by search | CPU | 5.01 | ❌ |
| K-5.08 | LPPLS filter conditions | `m ∈ [0.1, 0.9]`, `ω ∈ [6, 13]`, damping and oscillation-count checks | CPU | 5.07 | ❌ |
| K-5.09 | LPPLS batched GPU search | Grid `t_c × m × ω` × windows solved in chunks under 20 GB VRAM; float32 with centring; equivalence test against K-5.07 | GPU | 5.07, 1.05 | ❌ |
| K-5.10 | Float64 refinement | Best GPU cells polished on CPU float64 | CPU×N | 5.09 | ❌ |
| K-5.11 | LPPLS confidence indicator | Share of qualified fits across shrinking windows (Sornette style) as a time series | GPU | 5.09, 5.08 | ❌ |
| K-5.12 | Multiple-testing control | Benjamini–Hochberg over metrics for "`δ` > 0" (D10) | CPU | 5.04 | ❌ |
| K-5.13 | Independence grouping | Metrics grouped by `source_group`; "confirmed" means ≥ 3 distinct groups (D10) | — | 5.12, 2.01 | ❌ |
| K-5.14 | Acceleration dial | Dial object: `δ`, `t_c` interval, interval narrowing, confirmation count, state | CPU | 5.13, 5.11 | ❌ |

### Phase 6: Vinge Horizon `h*`

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-6.01 | Rolling-origin backtest harness | Wraps Prophet `cross_validation(parallel="processes")` with no lookahead, using vintages | CPU×N | 3.03 | ✅ |
| K-6.02 | Log-unit error metrics | Absolute log error (factor error) and CRPS replace MAPE on `ln y` (D2) | CPU | 6.01 | ✅ |
| K-6.03 | `h*` definition | First horizon where smoothed error crosses tolerance (D3), plus a variant where the 90% band exceeds k e-folds | CPU | 6.02 | ✅ |
| K-6.04 | `h*(t)` series | `h*` recomputed at every monthly cutoff for every series (the largest CPU job; uses the fit cache) | CPU×N | 6.03, 1.07 | ✅ |
| K-6.05 | NeuralProphet `h*` cross-check | Same procedure with NeuralProphet quantiles | GPU | 6.03, 4.05 | ✅ |
| K-6.06 | Shrinkage test | Mann–Kendall and Theil–Sen slope on `h*(t)` with a significance level | CPU | 6.04 | ✅ |
| K-6.07 | Vinge dial | Dial object: current `h*`, its trend, state | CPU | 6.06 | ❌ |

### Phase 7: Jonas Gap `G`, `dG/dt`

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-7.01 | Gap series | `G(t) = ln C − ln H` normalised at a base year; docs say the level is relative (D8) | CPU | 2.21 | ✅ |
| K-7.02 | Gap velocity with uncertainty | `dG/dt` with a bootstrap over index construction and the leave-one-out components | CPU×N | 7.01 | ✅ |
| K-7.03 | Sustained-positive test | Run-length rule plus HAC (Newey–West) t-test on `dG/dt > 0` | CPU | 7.02 | ✅ |
| K-7.04 | Component attribution | Which H sub-series lags the most: the funders' view | CPU | 7.02 | ✅ |
| K-7.05 | Closing scenarios | H growth rate needed to bring `dG/dt` to 0, with interval | CPU | 7.02 | ✅ |
| K-7.06 | Jonas dial | Dial object: `dG/dt`, interval, run length, state | CPU | 7.03 | ❌ |

### Phase 8: Ethical governor and tripwires

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-8.01 | Dial state machine | Calm/danger states with hysteresis so dials don't flicker; property tests | — | 5.14, 6.07, 7.06, 4.09 | ✅ |
| K-8.02 | Jonas asymmetric loss | Alarm when `P(danger) > 1/(1+λ)` (D6); `λ` read from public YAML | — | 8.01 | ✅ |
| K-8.03 | `P(danger)` per dial | Probabilities taken from bootstrap and ensemble distributions, not point estimates | CPU | 8.02 | ✅ |
| K-8.04 | Rawls maximin | Decision taken in the worst plausible scenario of the ensemble; reported next to the expected value | CPU | 8.03 | ✅ |
| K-8.05 | Collingridge lead time | Months until the upper quantile crosses the threshold, with 90% interval, e.g. "31 months (18–52)" | CPU | 8.03 | ✅ |
| K-8.06 | Intervention-cost curve | Parametric cost-versus-lead-time curve with documented assumptions | — | 8.05 | ✅ |
| K-8.07 | Goodhart divergence detector | Gap between public and held-out metric trends flagged as a gaming signal | CPU | 2.08 | ✅ |
| K-8.08 | Tripwire aggregator and spec | Levels 0–4 from §8 in versioned `tripwires.yaml`; property tests cover every level boundary | — | 8.02, 5.13 | ❌ |
| K-8.09 | Löb independence report | Fails the run if any dial depends only on a single lab-controlled source | — | 8.08 | ❌ |

### Phase 9: Historical validation

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-9.01 | Backtest runner | Replays the full pipeline at past "as-of" dates using only data available then | CPU×N | 6.01 | ✅ |
| K-9.02 | World population | From 1960 data, recovers von Foerster's `t_c` ≈ Nov 2026; from 1960–80 data, sees `δ` fall toward 0 | CPU×N | 9.01, 5.04 | ✅ |
| K-9.03 | Dot-com bubble | LPPLS `t_c` compared with Sornette's published values for Nasdaq 2000 | GPU | 5.11 | ❌ |
| K-9.04 | Bitcoin 2017 | LPPLS run before the December 2017 peak; lead time recorded | GPU | 5.11 | ❌ |
| K-9.05 | COVID-19 early 2020 | Conditional forecasts with lockdown events show the curve bending; counterfactual scored | CPU, GPU | 4.03 | ❌ |
| K-9.06 | Moore's law false-alarm test | Stays at Level 0 throughout | CPU×N | 8.08 | ❌ |
| K-9.07 | AI history replay | Changepoints and dial states from 2012 to today, as-of each year | CPU×N | 9.01, 8.08 | ❌ |
| K-9.08 | Scoring module | CRPS, PIT histograms, interval coverage, false-alarm rate, warning lead time | CPU | 6.02 | ❌ |
| K-9.09 | Validation report | `docs/validation.md` with figures and an honest list of failures | — | 9.02–9.08 | ❌ |

### Phase 10: Pre-registration ledger

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-10.01 | Forecast artifact and hashing | Canonical JSON (sorted keys, fixed float format) + SHA-256; the same forecast always gives the same hash | — | 8.08 | ✅ |
| K-10.02 | Hash-chained ledger | `ledger/forecasts.jsonl`, append-only; each entry includes the previous entry's hash and the data manifest hash (D11) | — | 10.01, 2.04 | ✅ |
| K-10.03 | Conditional forecast format | Stored as "If no new intervention, X reaches Y by Z (90% interval)" plus the event set assumed | — | 10.01 | ✅ |
| K-10.04 | External timestamping | Each weekly entry anchored with OpenTimestamps; proof files committed (D11) | — | 10.02 | ✅ |
| K-10.05 | Scorer for matured forecasts | Scores forecasts once their outcome date passes; interventions logged since then enable counterfactual scoring | CPU | 10.03, 9.08 | ❌ |
| K-10.06 | `kairos ledger verify` | Recomputes the chain and checks every proof; tamper tests (edited, deleted, reordered entries) all fail loudly | — | 10.04 | ❌ |
| K-10.07 | Public scorecard | Running calibration and CRPS table generated from the ledger | — | 10.05 | ❌ |

### Phase 11: Public dashboard

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-11.01 | Streamlit skeleton | `kairos dashboard` starts the app and reads only published artifacts | — | 0.05 | ✅ |
| K-11.02 | KAIROS Clock and tripwire banner | Lead time with 90% interval; current level and its recommended response | — | 8.05, 8.08 | ✅ |
| K-11.03 | Four dial widgets | Acceleration, Recursion, Vinge Horizon, Jonas Gap, each with state and sparkline | — | 8.01 | ✅ |
| K-11.04 | C versus H chart | Both curves always shown together (limitations table, §13) | — | 7.01 | ✅ |
| K-11.05 | Changepoint and acceleration explorer | Per-series Prophet fit, changepoints, `δ` bootstrap distribution | — | 3.04, 5.04 | ✅ |
| K-11.06 | Vinge Horizon chart | `h*(t)` with trend line and tolerance control | — | 6.04 | ✅ |
| K-11.07 | Ledger browser | Every forecast, its hash, proof status and score | — | 10.06 | ❌ |
| K-11.08 | Methodology page | `λ`, thresholds, assumptions and the D1–D12 notes in plain language | — | 8.08 | ❌ |
| K-11.09 | Static export | Plotly HTML snapshot published to GitHub Pages each week | — | 11.02–11.07 | ❌ |

### Phase 12: Automation, release, paper

| ID | Subtask | Done when | HW | Deps | Status |
|---|---|---|:-:|---|:-:|
| K-12.01 | Weekly pipeline command | `kairos run weekly`: refresh → fit → dials → governor → ledger → dashboard export | CPU×N, GPU | 10.04, 11.09 | ✅ |
| K-12.02 | Weekly GitHub Actions job | Cron job that runs the CPU-only pipeline and commits ledger hashes | CPU | 12.01 | ✅ |
| K-12.03 | Self-hosted GPU runner | Documented setup for this workstation as a labelled runner for GPU and `slow` jobs | GPU | 12.02 | ✅ |
| K-12.04 | Documentation site | `mkdocs-material` + `mkdocstrings` API reference, deployed to Pages | — | 0.04 | ✅ |
| K-12.05 | Tutorial notebooks | Three notebooks: null model, singularity detectors, reading the clock | CPU | 12.04 | ✅ |
| K-12.06 | Beyond AI example | Pandemic growth against health-system capacity, plus Scheffer critical-slowing-down signals as an optional fifth dial | CPU | 9.05 | ✅ |
| K-12.07 | Replication container | CUDA Docker image that reproduces the validation report end to end | GPU | 9.09 | ❌ |
| K-12.08 | Release v0.1.0 | `CHANGELOG.md`, version tag, PyPI trusted-publishing workflow | — | 12.04 | ❌ |
| K-12.09 | Paper draft | *"Measuring the Distance to a Singularity: Forecastability as an Early-Warning Signal"* in `paper/`, built in CI | — | 9.09, 10.07 | ❌ |

---

## 7. Definitions used in the tables

| Term | Meaning |
|---|---|
| `C(t)` | Composite AI capability index (K-2.21) |
| `H(t)` | Composite human oversight-capacity index (K-2.21) |
| `g(t)` | Growth rate `d ln x / dt`, per year |
| `δ` | Hyperbolicity exponent in `ln g = ln k + δ ln x`; `δ > 0` sustained means a finite-time singularity |
| `t_c` | Critical time: the date the hyperbolic or LPPLS fit blows up |
| `κ` | Recursion coefficient: out-of-sample skill gained from capability's own lags after exogenous drivers |
| `h*` | Vinge Horizon: furthest forecast horizon still inside the error tolerance |
| `G`, `dG/dt` | Jonas Gap and its velocity |
| `λ` | Jonas loss ratio: cost of a missed catastrophe relative to a false alarm |
| Dial | One of the four indicators; each is `calm` or `danger` |
| Tripwire level | 0 Normal, 1 Watch, 2 Warning, 3 Alarm, 4 Kairos (idea doc §8) |
| Source group | Data provenance cluster; metrics in different groups count as independent |
| Vintage | The version of a dataset as it existed on a given date; used to stop lookahead |

---

*KAIROS recommends; humans decide.*
