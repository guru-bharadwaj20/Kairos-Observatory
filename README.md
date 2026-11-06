# KAIROS Observatory

> *"A seismograph for the intelligence explosion, built to be proven wrong."*

[![CI](https://github.com/guru-bharadwaj20/Kairos-Observatory/actions/workflows/ci.yml/badge.svg)](https://github.com/guru-bharadwaj20/Kairos-Observatory/actions/workflows/ci.yml)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](.python-version)

KAIROS (Greek for *the decisive moment*) is an open, independent observatory that
watches two curves, **what AI can do**, `C(t)`, and **what humans can oversee**,
`H(t)`, and keeps answering one question:

> **How much time is left while intervention is still cheap?**

Prophet and NeuralProphet act as the null model of "normal" progress, mathematical
singularity detectors look for super-exponential acceleration, and a philosophy
layer turns ethical principles into explicit decision rules with published
parameters. Every forecast is hashed and pre-registered so anyone can score it.
KAIROS is designed as a **self-defeating prophecy**: it succeeds when its warnings
trigger the action that makes them wrong.

The full concept is in [KAIROS_Project_Idea.md](KAIROS_Project_Idea.md); the
build plan, rules and progress tracker are in [CONTRIBUTING.md](CONTRIBUTING.md).

## The four dials

| Dial | Symbol | What it measures | Danger sign |
|---|---|---|---|
| **Acceleration** | `δ`, `t_c` | Whether growth is super-exponential, and when it would blow up | `δ` significantly above 0 across several independent metrics, `t_c` interval narrowing |
| **Recursion** | `κ` | How much AI progress is driven by AI itself | `κ` rising across rolling windows |
| **Vinge Horizon** | `h*` | How far ahead we can still predict | `h*` shrinking |
| **Jonas Gap** | `G`, `dG/dt` | Capability outrunning comprehension | `dG/dt > 0`, sustained |

## Tripwire levels

| Level | Trigger | Recommended response |
|---|---|---|
| **0: Normal** | All dials calm | Weekly publication |
| **1: Watch** | One dial in the danger zone | More data collection, public notice |
| **2: Warning** | Two dials, or one confirmed on 3+ independent metrics | Mandatory third-party evaluation before new frontier deployments |
| **3: Alarm** | Three dials | Pause scaling past the current frontier until `H` catches up |
| **4: Kairos** | All four dials | The last window in which intervention is still cheap |

**KAIROS recommends; humans decide.** It is an instrument, not a kill switch.

## Quickstart

KAIROS pins **Python 3.12** (NeuralProphet 0.9 does not support 3.13) and uses
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/guru-bharadwaj20/Kairos-Observatory.git
cd Kairos-Observatory
uv python install 3.12
uv sync --all-extras          # creates .venv, installs CUDA torch on Windows/Linux
uv run kairos doctor          # checks CPU, RAM, CUDA, CmdStan
uv run pytest -m "not slow and not gpu"
```

Configuration comes from environment variables with the `KAIROS_` prefix (for
example `KAIROS_N_JOBS=8`, `KAIROS_DEVICE=cpu`) or a `kairos.yaml` file.

## Repository layout

```
src/kairos/
  cli.py           Typer entry point; auto-discovers kairos.commands.*
  settings.py      pydantic-settings configuration (KAIROS_* env, YAML)
  logging.py       rich logging with a run ID on every line
  types.py         SeriesFrame convention, SeriesMeta, DialReading
  compute/         thread governance, pmap, device selection, seeds, fit cache
  data/            loaders, vintages, schemas (data/sources/ per provider)
  models/          Prophet null models
  neural/          NeuralProphet and the recursion dial κ
  accel/           δ, t_c and LPPLS singularity detectors
  vinge/           Vinge Horizon h*
  jonas/           Jonas Gap G, dG/dt
  tripwires/       ethical governor and tripwire levels
  philosophy/      published decision rules (λ, maximin)
  validation/      historical backtests
  ledger/          hash-chained pre-registration ledger
  dashboard/       public dashboard
  commands/        one module per CLI subcommand
tests/             pytest suite (markers: slow, gpu)
scripts/           benchmarks and maintenance scripts
docs/              documentation and hardware benchmarks
```

## Hardware notes

The reference workstation is an Intel i9-14900 (8 P + 16 E cores, 32 threads),
128 GB RAM and an NVIDIA RTX 4500 Ada (24 GB). The CPU does the many small
Prophet/Stan fits through a 28-worker process pool with BLAS pinned to one
thread per worker; the GPU handles batched float32 work (LPPLS grids,
bootstraps, NeuralProphet ensembles) under a 20 GB VRAM budget. Measured
throughput lives in [docs/benchmarks.md](docs/benchmarks.md). See section 4 of
[CONTRIBUTING.md](CONTRIBUTING.md) for the full plan.

## Licence

Code is licensed under [Apache-2.0](LICENSE). KAIROS-built datasets and published
forecasts are licensed under [CC BY 4.0](LICENSE-DATA). Third-party data keeps
its original licence.
