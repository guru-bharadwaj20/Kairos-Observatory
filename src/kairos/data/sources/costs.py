"""Cost per unit of capability (``c.cost_per_mtok``).

The series is the cheapest blended API price (USD per million tokens, input and
output weighted 3:1) of any model whose MMLU score is at least
:data:`MMLU_THRESHOLD` (GPT-4 level at launch), kept at the months where a new
minimum is set. Falling cost means rising capability, so the composite index
uses it with direction ``-1`` (stored in the dataset's extra metadata).
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from kairos.data.schema import make_series, month_start
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.sources.metr import read_curated_csv
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = ["COSTS", "COST_PER_MTOK", "MMLU_THRESHOLD", "blended_price", "cost_at_threshold"]

MMLU_THRESHOLD: Final = 0.82
INPUT_WEIGHT: Final = 0.75

COST_PER_MTOK: Final = SeriesMeta(
    series_id="c.cost_per_mtok",
    name="Cheapest price per million tokens at MMLU >= 0.82",
    unit="USD/Mtok",
    source="costs",
    source_group="provider_pricing",
    side="C",
    transform="log",
    license="Facts from public pricing pages",
    url="https://epoch.ai/data-insights/llm-inference-price-trends",
)


def blended_price(input_usd: pd.Series, output_usd: pd.Series) -> pd.Series:
    """Blend input and output prices 3:1 (the usual chat workload mix)."""
    return INPUT_WEIGHT * input_usd + (1 - INPUT_WEIGHT) * output_usd


def cost_at_threshold(
    prices: pd.DataFrame, *, score: str = "mmlu", threshold: float = MMLU_THRESHOLD
) -> pd.DataFrame:
    """Running minimum price among models at or above a benchmark score.

    Args:
        prices: Columns ``model``, ``release_date``, ``score``,
            ``input_usd_per_mtok`` and ``output_usd_per_mtok``.
        score: Score column to threshold on.
        threshold: Minimum score for a model to count.

    Returns:
        Columns ``ds``, ``price``, ``model`` at each month a new minimum is set.
    """
    df = prices[prices[score] >= threshold].copy()
    df["price"] = blended_price(df["input_usd_per_mtok"], df["output_usd_per_mtok"])
    df["ds"] = month_start(df["release_date"]).to_numpy()
    df = df.sort_values(["ds", "price"], ascending=[True, False]).groupby("ds").last()
    df = df.reset_index()
    prev = df["price"].cummin().shift(fill_value=float("inf"))
    return df.loc[df["price"] < prev, ["ds", "price", "model"]].reset_index(drop=True)


def _build(ctx: BuildContext) -> BuildResult:
    prices = read_curated_csv(ctx.files["model_prices.csv"])
    hist = cost_at_threshold(prices)
    frame = make_series(
        COST_PER_MTOK,
        hist["ds"],
        hist["price"],
        vintage=str(ctx.vintage),
        extra={"model": hist["model"].tolist()},
    )
    extra = {"direction": -1, "threshold": {"mmlu": MMLU_THRESHOLD}, "blend": "3:1 in:out"}
    return BuildResult(datasets=[Dataset(COST_PER_MTOK, frame, extra)])


COSTS: Final = register(
    Source(
        name="costs",
        title="Model API list prices",
        license="Facts from public pricing pages",
        url="https://epoch.ai/data-insights/llm-inference-price-trends",
        files=(RawFile("model_prices.csv"),),
        build=_build,
    )
)
