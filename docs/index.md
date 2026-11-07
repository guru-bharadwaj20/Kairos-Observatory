# KAIROS Observatory

> *A seismograph for the intelligence explosion, built to be proven wrong.*

KAIROS is an open early-warning instrument. It watches two curves, what AI can
do, `C(t)`, and what humans can oversee, `H(t)`, and keeps answering one
question: **how much time is left while intervention is still cheap?**

It reports four dials, a tripwire level agreed in advance, and a lead time with
a 90% interval. Every weekly forecast is hashed into a public, timestamped
ledger before its outcome is known.

| Dial | Symbol | Danger sign |
|---|---|---|
| Acceleration | `δ`, `t_c` | `δ` significantly above 0 on several independent metrics |
| Recursion | `κ` | `κ` rising across rolling windows |
| Vinge Horizon | `h*` | `h*` shrinking |
| Jonas Gap | `dG/dt` | `dG/dt > 0`, sustained |

**KAIROS recommends; humans decide.**

## Where to go next

- [Methodology](methodology.md): the mathematics and the decision rules.
- [Weekly pipeline](weekly.md): what `kairos run weekly` does.
- [API reference](reference/run.md): generated from the docstrings.
- [Design notes](design.md): the non-negotiables, toolchain pins, hardware
  plan and design notes D1–D12.
- The full idea lives in the repository as
  [`KAIROS_Project_Idea.md`](https://github.com/guru-bharadwaj20/Kairos-Observatory/blob/main/KAIROS_Project_Idea.md).
