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
