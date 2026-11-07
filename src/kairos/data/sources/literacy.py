"""H: public AI literacy (``h.ai_literacy``).

Survey points (Pew: share of U.S. adults who have used ChatGPT) are interpolated
linearly to months. Uncertainty **widens between surveys**: the variance at month
``t`` between surveys at ``a`` and ``b`` is the interpolated survey variance plus a
Brownian-bridge term ``tau**2 * (t - a) * (b - t) / (b - a)``. Bands are 90%.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from kairos.data.schema import make_series, month_start
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.sources.metr import read_curated_csv
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = ["AI_LITERACY", "LITERACY", "TAU", "interpolate_surveys"]

TAU: Final = 0.01
"""Drift scale of the bridge, in share points per sqrt(month)."""

AI_LITERACY: Final = SeriesMeta(
    series_id="h.ai_literacy",
    name="U.S. adults who have used ChatGPT (Pew)",
    unit="fraction",
    source="pew",
    source_group="pew",
    side="H",
    transform="logit",
    license="Pew Research Center terms of use (facts cited)",
    url="https://www.pewresearch.org/topic/internet-technology/emerging-technology/artificial-intelligence/",
)


def interpolate_surveys(
    points: pd.DataFrame, *, tau: float = TAU, z: float = 1.645
) -> pd.DataFrame:
    """Interpolate survey shares monthly with uncertainty that widens between surveys.

    Args:
        points: Columns ``date``, ``value`` (share) and ``se`` (standard error).
        tau: Bridge drift scale (share per sqrt(month)).
        z: Normal quantile of the band (1.645 = 90%).

    Returns:
        Columns ``ds``, ``y``, ``y_lo``, ``y_hi``, ``sd``, ``interpolated``.
    """
    p = points.assign(ds=month_start(points["date"]).to_numpy()).sort_values("ds")
    months = pd.date_range(p["ds"].iloc[0], p["ds"].iloc[-1], freq="MS")
    t = np.arange(len(months), dtype=float)
    x = months.get_indexer(pd.DatetimeIndex(p["ds"])).astype(float)
    y = np.interp(t, x, p["value"].to_numpy(dtype=float))
    var = np.interp(t, x, p["se"].to_numpy(dtype=float) ** 2)
    right = np.clip(np.searchsorted(x, t, side="left"), 1, len(x) - 1)
    a, b = x[right - 1], x[right]
    bridge = tau**2 * np.clip((t - a) * (b - t), 0, None) / np.maximum(b - a, 1.0)
    sd = np.sqrt(var + bridge)
    return pd.DataFrame(
        {
            "ds": months.astype("datetime64[ns]"),
            "y": y,
            "y_lo": np.clip(y - z * sd, 0.0, y),
            "y_hi": np.clip(y + z * sd, y, 1.0),
            "sd": sd,
            "interpolated": ~np.isin(t, x),
        }
    )


def _build(ctx: BuildContext) -> BuildResult:
    pts = read_curated_csv(ctx.files["ai_literacy_surveys.csv"])
    m = interpolate_surveys(pts)
    frame = make_series(
        AI_LITERACY, m["ds"], m["y"], vintage=ctx.vintage, y_lo=m["y_lo"], y_hi=m["y_hi"],
        extra={"interpolated": m["interpolated"].tolist()},
    )  # fmt: skip
    extra = {"interpolation": "linear + Brownian-bridge band", "tau": TAU, "band": "90%"}
    return BuildResult(datasets=[Dataset(AI_LITERACY, frame, extra)])


LITERACY: Final = register(
    Source(
        name="literacy",
        title="Pew Research Center surveys on AI use",
        license="Pew Research Center terms of use (facts cited)",
        url=AI_LITERACY.url,
        files=(RawFile("ai_literacy_surveys.csv"),),
        build=_build,
    )
)
