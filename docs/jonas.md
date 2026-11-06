# Jonas Gap `G` and `dG/dt`

> The danger is not a level of capability but a gap: capability `C(t)` growing
> faster than humanity's capacity `H(t)` to understand, evaluate and govern it.

Code: `kairos.jonas`. Inputs are `SeriesFrame`s (see `kairos.types`) whose
`source_group` column marks each row as `"C"` (capability) or `"H"` (human
capacity). The frame may hold the composite indices from K-2.21 or their raw
components.

## Gap series (K-7.01)

```
G(t) = ln C(t) − ln H(t)
```

**The level of `G` is relative, not absolute** (design note D8). `C` and `H` are
in different units, so `ln C − ln H` depends on an arbitrary choice of units. Each
component is divided by its own mean over a **base year** before logging, and each
side is the weighted geometric mean of its normalised components (weights are
renormalised over the components observed in a given month). `G` is therefore 0 in
the base year. `G = 0.7` means "since the base year, capability has grown by a
factor of e^0.7 ≈ 2 more than human capacity", and nothing about whether the two are
"balanced". Changing the base year shifts `G` by a constant and leaves `dG/dt`
unchanged. **`dG/dt` is the headline quantity.**

```python
from kairos.jonas import jonas_gap
gap = jonas_gap(frame, base_year=2016)   # ds, ln_c, ln_h, gap, n_c, n_h
```

## Gap velocity `dG/dt` (K-7.02)

`dG/dt` at month `t` is the OLS slope of `G` over the trailing `window` months
(default 12), in log units per year. It never uses data after `t`.

Uncertainty, from `gap_velocity(frame, base_year, n_boot=500)`:

| Column | Meaning |
|---|---|
| `boot_lo`, `boot_hi`, `boot_se` | Bootstrap over index construction: components redrawn with replacement within each side, and observations perturbed log-normally so their 90% interval matches `y_lo`/`y_hi` |
| `loo_min`, `loo_max` | Range of `dG/dt` over leave-one-out composites (each component dropped in turn) |
| `dgdt_lo`, `dgdt_hi` | Reported interval: the union of the two |

Replicates run on `kairos.compute.pmap`; each one is seeded with
`derive_seed(seed, "jonas.velocity", r)`, so results are identical for any `n_jobs`.

## Sustained-positive test (K-7.03)

The danger sign is `dG/dt > 0` *sustained*. `sustained_positive(velocity)` requires
both:

1. **Run length:** trailing `dG/dt` positive for at least `min_run` months (default 6).
2. **HAC t-test:** over the last `months` months (default 36), the annualised
   increments `12·ΔG` have a mean above 0 at one-sided level `alpha` (default 0.05),
   using Newey–West standard errors with `floor(4·(n/100)^(2/9))` lags. Composite
   indices are smoothed and interpolated, so the increments are autocorrelated and a
   plain t-test would overstate significance.
