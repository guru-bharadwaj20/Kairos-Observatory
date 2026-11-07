# Acceleration dial: `δ`, `t_c` and LPPLS (Phase 5)

Code lives in `src/kairos/accel/`. Time is always in decimal years.

| Module | Task | What it does |
|---|---|---|
| `synthetic` | K-5.01 | Exponential, hyperbolic and logistic test series with known `δ`, `t_c` |
| `growth` | K-5.02 | Smoothing-spline `g = d ln x/dt` on history only (GCV, γ = 1.4); `ln g = ln k + δ ln x`; `T_remaining = 1/(δ·g)` with delta-method interval |
| `hyperbolic` | K-5.03 | Direct fit of `ln x = a − (1/δ) ln(t_c − t)` with `scipy.optimize.least_squares` |
| `bootstrap` | K-5.04 | Residual moving block bootstrap, smoother refit per draw, CPU pool |
| `bootstrap_gpu` | K-5.05 | Same estimator vectorised in torch (10⁵ draws, float32 on GPU) |
| `coverage` | K-5.06 | Monte Carlo coverage of the 90 % `δ` intervals |
| `lppls` | K-5.07 | Float64 LPPLS reference: OLS for `A, B, C1, C2`, grid + Nelder–Mead for `t_c, m, ω` |
| `lppls_filter` | K-5.08 | `m ∈ [0.1, 0.9]`, `ω ∈ [6, 13]`, damping ≥ 0.8, oscillations ≥ 2.5, `B < 0`, `t_c` near window end |
| `lppls_gpu` | K-5.09 | Batched `t_c × m × ω` × windows grid in torch, centred and column-scaled float32 |
| `lppls_refine` | K-5.10 | Float64 polishing of the best grid cells on the CPU pool |
| `lppls_confidence` | K-5.11 | Share of qualified fits over shrinking windows, per end date |
| `multitest` | K-5.12 | Benjamini–Hochberg over metrics for `δ > 0` |
| `independence` | K-5.13 | Confirmation needs ≥ 3 distinct `source_group`s |
| `dial` | K-5.14 | `acceleration_dial(...) -> DialReading` with state ladder 0–4 |

CLI: `kairos accel delta FILE [--series-id ID] [--n-boot N] [--device cuda]`.

## Precision and equivalence

GPU kernels run in float32; every one has a float64 reference and a test:
`bootstrap_gpu` vs `bootstrap` (two-sample KS on `δ*` and `t_c*`), and `lppls_gpu`
vs `lppls.lppls_sse_cells` (full SSE grid within 10⁻⁴ of the window's total sum of
squares, and the GPU argmin cell is optimal in float64 to the same tolerance).

## Known limitation: interval coverage

Measured 90 % `δ` interval coverage (400 trials, 240 monthly points):

| Scenario | Coverage |
|---|--:|
| `δ = 1.0`, iid log noise 0.1 | 0.87 |
| `δ = 0.5`, iid log noise 0.1 | 0.84 |
| `δ = 0.5`, iid log noise 0.05 | 0.74 |
| `δ = 0.5`, AR(1) 0.3 noise 0.1 | 0.68 |

The bootstrap does not reproduce the sample-to-sample variation of the GCV penalty,
and autocorrelated noise makes GCV undersmooth, so intervals are somewhat narrow.
The BH step and the three-group confirmation rule guard the dial against this.
