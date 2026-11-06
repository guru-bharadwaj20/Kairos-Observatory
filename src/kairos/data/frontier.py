"""Frontier envelope: irregular model releases to a monthly frontier series (D7).

A plain running maximum of per-model values produces long flat steps that Prophet
reads as many changepoints. The envelope here instead

1. reduces releases to **record points**: months where the frontier statistic
   (running maximum, or running mean of the top-``k`` values in log space) rises;
2. joins consecutive record points by **log-linear interpolation** (or a step, if
   asked), so growth between records is spread over the gap;
3. flags every interpolated month and every month inside a gap longer than
   ``max_gap_months`` (``gap=True``), so models can down-weight them.

The method and its parameters are returned for storage in the series metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from kairos.data.schema import month_start

__all__ = ["EnvelopeSpec", "frontier_envelope"]

Method = Literal["running_max", "top_k"]
Interp = Literal["loglinear", "step"]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EnvelopeSpec:
    """Parameters of :func:`frontier_envelope`.

    Attributes:
        method: ``"running_max"`` or ``"top_k"`` (geometric mean of the ``k`` largest
            values released so far).
        k: Number of models in the top-``k`` mean.
        interpolation: ``"loglinear"`` between record points, or ``"step"``.
        max_gap_months: Months between records above which the span is flagged.
        start: Ignore releases before this date (``None`` keeps all).
    """

    method: Method = "top_k"
    k: int = 3
    interpolation: Interp = "loglinear"
    max_gap_months: int = 12
    start: str | None = "2010-01-01"

    def as_metadata(self) -> dict[str, Any]:
        """Return the spec as a JSON-ready dict tagged with the builder name."""
        return {"builder": "kairos.data.frontier.frontier_envelope", **asdict(self)}


def _statistic(values: FloatArray, spec: EnvelopeSpec) -> FloatArray:
    """Running frontier statistic in log space after each release (sorted by date)."""
    logs = np.log(values)
    if spec.method == "running_max":
        return np.asarray(np.maximum.accumulate(logs), dtype=float)
    out = np.empty_like(logs)
    best: list[float] = []
    for i, v in enumerate(logs):
        best.append(float(v))
        best.sort(reverse=True)
        del best[spec.k :]
        out[i] = float(np.mean(best))
    return out


def frontier_envelope(
    releases: pd.DataFrame,
    spec: EnvelopeSpec | None = None,
    *,
    value: str = "value",
) -> pd.DataFrame:
    """Build a monthly frontier series from irregular releases.

    Args:
        releases: One row per release with columns ``ds`` (any date) and ``value``
            (strictly positive; non-positive or missing values are dropped).
        spec: Envelope parameters (defaults to :class:`EnvelopeSpec`).
        value: Name of the value column.

    Returns:
        Monthly frame from the first to the last record month with columns
        ``ds`` (month start), ``y``, ``record`` (a record was set this month),
        ``interpolated``, ``gap`` and ``months_since_record``.

    Raises:
        ValueError: If no usable release remains.
    """
    spec = spec or EnvelopeSpec()
    df = releases[["ds", value]].rename(columns={value: "v"}).copy()
    df["ds"] = month_start(df["ds"]).to_numpy()
    df = df[np.isfinite(df["v"]) & (df["v"] > 0)]
    if spec.start is not None:
        df = df[df["ds"] >= pd.Timestamp(spec.start)]
    if df.empty:
        raise ValueError("no positive releases to build an envelope from")
    df = df.sort_values(["ds", "v"], kind="stable")
    stat = _statistic(df["v"].to_numpy(dtype=float), spec)
    per_month = pd.Series(stat, index=df["ds"].to_numpy()).groupby(level=0).last()
    rising = per_month.to_numpy() > np.maximum.accumulate(np.r_[-np.inf, per_month.to_numpy()[:-1]])
    records = per_month[rising]

    months = pd.date_range(records.index[0], records.index[-1], freq="MS")
    idx = np.arange(len(months))
    rec_pos = months.get_indexer(records.index)
    if spec.interpolation == "loglinear":
        log_y = np.interp(idx, rec_pos, records.to_numpy())
    else:
        log_y = records.to_numpy()[np.searchsorted(rec_pos, idx, side="right") - 1]
    is_record = np.isin(idx, rec_pos)
    last_rec = rec_pos[np.searchsorted(rec_pos, idx, side="right") - 1]
    next_rec = rec_pos[np.minimum(np.searchsorted(rec_pos, idx, side="left"), len(rec_pos) - 1)]
    return pd.DataFrame(
        {
            "ds": months.astype("datetime64[ns]"),
            "y": np.exp(log_y),
            "record": is_record,
            "interpolated": ~is_record,
            "gap": (~is_record) & ((next_rec - last_rec) > spec.max_gap_months),
            "months_since_record": (idx - last_rec).astype("int64"),
        }
    )
