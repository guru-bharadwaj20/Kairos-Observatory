# Changelog

All notable changes to KAIROS Observatory are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-13

First public release: the open dataset, the four dials, the ethical governor,
the pre-registration ledger and the public dashboard.

### Added

- Foundation: `uv` project pinned to Python 3.12, lint, types, tests, CI,
  governance and independence policy, data licence, citation metadata.
- Compute layer for the i9-14900 / RTX 4500 Ada workstation: thread
  governance, `pmap` executor, device selection, reproducible seeding, fit cache.
- Open dataset for capability `C(t)`, human capacity `H(t)` and the event
  calendar, with vintages and a data manifest.
- Prophet null models on `ln y` and changepoint analysis.
- NeuralProphet models and the recursion dial `κ` with a permutation null.
- Singularity detectors: hyperbolic `δ` fit, `t_c`, block bootstrap, LPPLS on
  the GPU with a float64 CPU reference.
- Vinge Horizon `h*(t)` in log-unit error metrics.
- Jonas Gap `G` and `dG/dt` with uncertainty.
- Ethical governor: Jonas asymmetric loss, Rawls maximin, Collingridge lead
  time, Goodhart divergence, Löb independence check, tripwire levels 0–4.
- Historical validation: world population, dot-com, Bitcoin 2017, COVID-19,
  Moore's law false-alarm test, AI history replay.
- Hash-chained, OpenTimestamps-anchored forecast ledger and public scorecard.
- Streamlit dashboard and weekly static export to GitHub Pages.
- `kairos run weekly` and `kairos run validation` orchestrators.
- Weekly CPU-only GitHub Actions job; self-hosted GPU runner workflow.
- Documentation site (mkdocs-material, mkdocstrings) and three tutorial
  notebooks.
- Beyond-AI example: pandemic capacity gap and a critical-slowing-down dial.
- CUDA replication container for the validation report.
- Paper draft, built in CI.

[Unreleased]: https://github.com/guru-bharadwaj20/Kairos-Observatory/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/guru-bharadwaj20/Kairos-Observatory/releases/tag/v0.1.0
