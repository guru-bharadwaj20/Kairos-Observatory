"""Composite indices ``C(t)`` (``idx.C``) and ``H(t)`` (``idx.H``).

Method (``geomean``, the default):

1. Each component ``y_i`` is put on its model scale ``z_i`` (``log`` or ``logit``,
   from :class:`~kairos.types.SeriesMeta.transform`; logits use a 2% clip so zero
   shares do not explode) and interpolated linearly to months between its first
   and last observation. Nothing is extrapolated.
2. Monthly growth ``g_i = Δz_i`` is **normalised** by its own standard deviation
   ``s_i`` and multiplied by the component's direction (``-1`` for cost or lag
   series, read from ``extra["direction"]``).
3. The composite growth averages the available normalised growths and maps them
   back to log units, ``G = Σ g_i/s_i / Σ 1/s_i`` (a precision-weighted mean), and
   the index is ``exp(cumsum G)``: a **geometric mean** of component indices.
   It is defined from the first to the last month with at least
   ``min_components`` components and rebased to 1 at ``base`` (D8: only relative
   levels and ``dG/dt`` mean anything).

Variant ``pca``: the composite growth is the first principal component of the
normalised growths (missing entries set to 0, the standardised mean), signed so
that its loadings sum to a positive number.

:func:`leave_one_out` reports how much each component moves the index.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from kairos.data.schema import make_series
from kairos.data.store import Dataset
from kairos.data.transforms import logit
from kairos.types import SeriesMeta

__all__ = [
    "C_COMPONENTS",
    "H_COMPONENTS",
    "INDEX_C",
    "INDEX_H",
    "CompositeResult",
    "build_indices",
    "composite_index",
    "leave_one_out",
]

Method = Literal["geomean", "pca"]
LOGIT_EPS: Final = 0.02
SCALE_FLOOR: Final = 1e-3
"""Smallest growth scale, so a near-constant component cannot dominate."""
BASE: Final = "2020-01-01"

C_COMPONENTS: Final = (
    "c.frontier_compute",
    "c.metr_horizon",
    "c.bench.mmlu",
    "c.bench.gpqa_diamond",
    "c.bench.swe_bench_verified",
    "c.cost_per_mtok",
    "c.ai_research_share",
)
H_COMPONENTS: Final = (
    "h.safety_papers",
    "h.safety_researchers",
    "h.safety_funding",
    "h.eval_share",
    "h.regulatory_lag",
    "h.ai_literacy",
)


def _index_meta(side: Literal["C", "H"]) -> SeriesMeta:
    what = "AI capability" if side == "C" else "human oversight capacity"
    return SeriesMeta(
        series_id=f"idx.{side}",
        name=f"Composite {what} index {side}(t)",
        unit="index (base=1)",
        source="kairos",
        source_group="composite",
        side=side,
        transform="log",
        license="CC-BY-4.0",
        url="https://github.com/guru-bharadwaj20/Kairos-Observatory",
    )


INDEX_C: Final = _index_meta("C")
INDEX_H: Final = _index_meta("H")


@dataclass(frozen=True, slots=True)
class CompositeResult:
    """A composite index and how it was built.

    Attributes:
        index: Monthly ``ds``, ``y`` (index level), ``growth``, ``n_components``.
        components: Component IDs used.
        weights: Component -> weight in the composite growth (PCA loadings or 1/n).
        scale: Component -> growth standard deviation used for normalisation.
    """

    index: pd.DataFrame
    components: tuple[str, ...]
    weights: dict[str, float]
    scale: dict[str, float]


def _monthly_scale(d: Dataset) -> pd.Series:
    f = d.frame
    y = f["y"].to_numpy(dtype=float)
    if d.meta.transform == "logit":
        z = logit(y, eps=LOGIT_EPS)
    elif d.meta.transform == "log":
        z = np.log(np.clip(y, np.finfo(float).tiny, None))
    else:
        z = y
    s = pd.Series(z, index=pd.DatetimeIndex(f["ds"])).groupby(level=0).mean()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(grid).interpolate(method="time", limit_area="inside")


def _growth_matrix(datasets: Mapping[str, Dataset], ids: Sequence[str]) -> tuple[
    pd.DataFrame, dict[str, float]
]:  # fmt: skip
    cols: dict[str, pd.Series] = {}
    scale: dict[str, float] = {}
    for sid in ids:
        d = datasets[sid]
        g = _monthly_scale(d).diff()
        sd = float(g.std())
        scale[sid] = max(sd, SCALE_FLOOR) if np.isfinite(sd) else 1.0
        direction = float(d.extra.get("direction", 1))
        cols[sid] = direction * g / scale[sid]
    return pd.DataFrame(cols), scale


def composite_index(
    datasets: Mapping[str, Dataset],
    components: Sequence[str],
    *,
    method: Method = "geomean",
    min_components: int = 2,
    base: str = BASE,
) -> CompositeResult:
    """Combine component series into one composite index.

    Args:
        datasets: Available datasets by ID; missing components are skipped.
        components: Candidate component IDs.
        method: ``"geomean"`` or ``"pca"``.
        min_components: Components required for a month to bound the index range.
        base: Rebase date (the first index month is used if ``base`` is outside).

    Raises:
        ValueError: If fewer than ``min_components`` components are available or
            no month has enough components.
    """
    ids = tuple(c for c in components if c in datasets and len(datasets[c].frame) >= 2)
    if len(ids) < min_components:
        raise ValueError(f"need {min_components} components, have {list(ids)}")
    growth, scale = _growth_matrix(datasets, ids)
    counts = growth.notna().sum(axis=1)
    ok = counts[counts >= min_components].index
    if ok.empty:
        raise ValueError("no month has enough components")
    g = growth.loc[ok.min() : ok.max()]
    if method == "geomean":
        inv = pd.Series({c: 1.0 / scale[c] for c in ids})
        avail = g.notna().astype(float) * inv
        # g already holds direction * growth / s; dividing by sum(1/s) maps back to log units
        combined = g.sum(axis=1, min_count=1) / avail.sum(axis=1).replace(0, np.nan)
        combined = combined.fillna(0.0)
        weights = {c: float(inv[c] / inv.sum()) for c in ids}
    else:
        x = g.fillna(0.0).to_numpy()
        _, _, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
        load = vt[0] * (1.0 if vt[0].sum() >= 0 else -1.0)
        load = load / np.abs(load).sum()
        ref = float(np.median(list(scale.values())))
        combined = pd.Series((x @ load) * ref, index=g.index)
        weights = dict(zip(ids, map(float, load), strict=True))
    level = combined.cumsum()
    base_ts = pd.Timestamp(base)
    if base_ts not in level.index:
        base_ts = level.index[0]
    level = level - level.loc[base_ts]
    index = pd.DataFrame(
        {
            "ds": level.index.astype("datetime64[ns]"),
            "y": np.exp(level.to_numpy()),
            "growth": combined.to_numpy(),
            "n_components": g.notna().sum(axis=1).to_numpy().astype("int64"),
        }
    )
    return CompositeResult(index, ids, weights, scale)


def leave_one_out(
    datasets: Mapping[str, Dataset], components: Sequence[str], **kwargs: Any
) -> pd.DataFrame:
    """Sensitivity of a composite to dropping each component in turn.

    Returns:
        One row per dropped component: ``dropped``, ``growth_per_year`` (mean
        normalised growth x 12), ``corr_growth`` with the full index and
        ``max_abs_log_dev`` of the level over the common months. The first row
        (``dropped = "(none)"``) is the full index.
    """
    full = composite_index(datasets, components, **kwargs)
    ref = full.index.set_index("ds")
    rows = [{"dropped": "(none)", "growth_per_year": 12 * ref["growth"].mean(),
             "corr_growth": 1.0, "max_abs_log_dev": 0.0}]  # fmt: skip
    for c in full.components:
        try:
            alt = composite_index(datasets, [x for x in full.components if x != c], **kwargs)
        except ValueError:
            continue
        a = alt.index.set_index("ds")
        common = ref.index.intersection(a.index)
        dev = np.log(a.loc[common, "y"]) - np.log(ref.loc[common, "y"])
        dev = dev - dev.iloc[0]
        rows.append(
            {
                "dropped": c,
                "growth_per_year": 12 * a["growth"].mean(),
                "corr_growth": float(ref.loc[common, "growth"].corr(a.loc[common, "growth"])),
                "max_abs_log_dev": float(np.abs(dev).max()),
            }
        )
    return pd.DataFrame(rows)


def build_indices(
    datasets: Mapping[str, Dataset],
    vintage: date,
    *,
    method: Method = "geomean",
    min_components: int = 1,
) -> tuple[list[Dataset], dict[str, pd.DataFrame]]:
    """Build ``idx.C`` and ``idx.H`` plus their leave-one-out tables.

    ``min_components`` defaults to 1 so the indices cover the full history of their
    longest component (``n_components`` in the frame shows the coverage).

    Returns:
        The two index datasets and tables ``loo_C`` and ``loo_H``.
    """
    out: list[Dataset] = []
    tables: dict[str, pd.DataFrame] = {}
    for meta, comps in ((INDEX_C, C_COMPONENTS), (INDEX_H, H_COMPONENTS)):
        res = composite_index(datasets, comps, method=method, min_components=min_components)
        frame = make_series(
            meta, res.index["ds"], res.index["y"], vintage=vintage,
            extra={"growth": res.index["growth"].tolist(),
                   "n_components": res.index["n_components"].tolist()},
        )  # fmt: skip
        extra = {"method": method, "components": list(res.components), "weights": res.weights,
                 "growth_scale": res.scale, "base": BASE, "logit_eps": LOGIT_EPS}  # fmt: skip
        out.append(Dataset(meta, frame, extra))
        tables[f"loo_{meta.side}"] = leave_one_out(
            datasets, comps, method=method, min_components=min_components
        )
    return out, tables
