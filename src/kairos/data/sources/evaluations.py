"""H: share of frontier releases with an independent pre-deployment evaluation.

``h.eval_share`` is, per calendar quarter (dated at the quarter's first month), the
fraction of Epoch-flagged frontier models released that quarter that had a publicly
documented third-party pre-deployment evaluation. ``y_lo``/``y_hi`` are 90% Wilson
score intervals, which are wide because each quarter holds only a few releases.
Quarters without frontier releases are omitted.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from kairos.data.schema import make_series
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.sources.epoch import MODELS_TABLE
from kairos.data.sources.metr import read_curated_csv
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = ["EVALUATIONS", "EVAL_SHARE", "evaluation_share", "wilson_interval"]

EVAL_SHARE: Final = SeriesMeta(
    series_id="h.eval_share",
    name="Share of frontier releases with third-party pre-deployment evaluation",
    unit="fraction",
    source="evaluations",
    source_group="evaluator_disclosures",
    side="H",
    transform="logit",
    license="Facts from public system cards and reports",
    url="https://evaluations.metr.org/",
)


def wilson_interval(k: Any, n: Any, z: float = 1.645) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64]
]:  # fmt: skip
    """Wilson score interval for ``k`` successes out of ``n`` (90% by default)."""
    kf = np.asarray(k, dtype=float)
    nf = np.asarray(n, dtype=float)
    p = kf / nf
    denom = 1 + z**2 / nf
    centre = (p + z**2 / (2 * nf)) / denom
    half = z * np.sqrt(p * (1 - p) / nf + z**2 / (4 * nf**2)) / denom
    lo = np.asarray(np.clip(centre - half, 0, 1), dtype=float)
    hi = np.asarray(np.clip(centre + half, 0, 1), dtype=float)
    return lo, hi


def evaluation_share(
    models: pd.DataFrame, evaluated: set[str], *, start: str = "2023-01-01"
) -> pd.DataFrame:
    """Per-quarter share of frontier releases with a third-party evaluation.

    Args:
        models: The ``epoch_models`` table (needs ``ds``, ``model``, ``frontier``).
        evaluated: Model names (as in ``models``) that were evaluated.
        start: First release date counted.

    Returns:
        Columns ``ds`` (quarter start), ``n``, ``k``, ``share``, ``lo``, ``hi``.
    """
    f = models[models["frontier"] & (models["ds"] >= pd.Timestamp(start))].copy()
    f["ds"] = f["ds"].dt.to_period("Q").dt.to_timestamp().astype("datetime64[ns]")
    f["hit"] = f["model"].isin(evaluated)
    g = f.groupby("ds").agg(n=("hit", "size"), k=("hit", "sum")).reset_index()
    g["share"] = g["k"] / g["n"]
    lo, hi = wilson_interval(g["k"].to_numpy(), g["n"].to_numpy())
    g["lo"], g["hi"] = np.minimum(lo, g["share"]), np.maximum(hi, g["share"])
    return g


def _build(ctx: BuildContext) -> BuildResult:
    evals = read_curated_csv(ctx.files["third_party_evals.csv"])
    share = evaluation_share(ctx.tables[MODELS_TABLE], set(evals["model"]))
    frame = make_series(
        EVAL_SHARE, share["ds"], share["share"], vintage=ctx.vintage,
        y_lo=share["lo"], y_hi=share["hi"],
        extra={"n_releases": share["n"].tolist(), "n_evaluated": share["k"].tolist()},
    )  # fmt: skip
    extra = {"interval": "Wilson 90%", "frontier_flag": "Epoch AI 'Frontier model'"}
    return BuildResult(datasets=[Dataset(EVAL_SHARE, frame, extra)])


EVALUATIONS: Final = register(
    Source(
        name="evaluations",
        title="Third-party pre-deployment evaluations of frontier models",
        license="Facts from public system cards and reports",
        url="https://evaluations.metr.org/",
        files=(RawFile("third_party_evals.csv"),),
        build=_build,
        requires=("epoch",),
    )
)
