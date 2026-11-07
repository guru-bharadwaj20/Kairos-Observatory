"""Multiple-testing control for "``δ > 0``" across metrics (K-5.12, design note D10).

Each metric gets a one-sided bootstrap p-value for ``H0: δ <= 0``
(:attr:`kairos.accel.bootstrap.BootstrapResult.p_delta_positive`). Testing many
metrics inflates false positives, so the p-values are adjusted with the
Benjamini–Hochberg (1995) step-up procedure, which controls the false discovery rate
at ``q`` under independence or positive dependence.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from kairos.accel.bootstrap import BootstrapResult

__all__ = ["benjamini_hochberg", "delta_significance"]


def benjamini_hochberg(
    p_values: npt.ArrayLike, q: float = 0.1
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Benjamini–Hochberg step-up procedure.

    Args:
        p_values: Raw p-values (``nan`` entries are never rejected and stay ``nan``).
        q: Target false discovery rate.

    Returns:
        ``(reject, p_adjusted)`` in the input order.
    """
    if not 0 < q < 1:
        raise ValueError("q must lie in (0, 1)")
    p = np.asarray(p_values, dtype=np.float64)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    adj = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = len(ok)
    if m:
        order = ok[np.argsort(p[ok], kind="stable")]
        ranked = p[order] * m / np.arange(1, m + 1)
        adj[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    reject = np.where(np.isfinite(adj), adj <= q, False)
    return reject, adj


def delta_significance(
    results: Mapping[str, BootstrapResult], *, q: float = 0.1, level: float = 0.9
) -> pd.DataFrame:
    """Per-metric ``δ``, interval, raw and BH-adjusted p-value for ``δ > 0``.

    Args:
        results: Series ID → bootstrap result.
        q: False discovery rate.
        level: Level of the reported ``δ`` interval.

    Returns:
        Frame with ``series_id``, ``delta``, ``delta_lo``, ``delta_hi``, ``p_value``,
        ``p_adj`` and ``significant``, sorted by ``p_value``.
    """
    rows = []
    for sid, res in results.items():
        lo, hi = res.interval(level)
        rows.append(
            {
                "series_id": sid,
                "delta": res.point.delta,
                "delta_lo": lo,
                "delta_hi": hi,
                "p_value": res.p_delta_positive,
            }
        )
    df = pd.DataFrame(rows, columns=["series_id", "delta", "delta_lo", "delta_hi", "p_value"])
    reject, adj = benjamini_hochberg(df["p_value"].to_numpy(), q)
    df["p_adj"] = adj
    df["significant"] = reject
    return df.sort_values("p_value", kind="stable").reset_index(drop=True)
