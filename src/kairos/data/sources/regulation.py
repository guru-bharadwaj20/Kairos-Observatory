"""H: regulatory lag (``h.regulatory_lag``).

For each capability, the lag in months from its first public appearance to the
first binding rule anywhere that addresses it, dated at the rule's month. Rules
are taken from the OECD.AI policy database (or the official texts it links).
A **longer** lag means weaker oversight, so the composite H index uses this
series with direction ``-1``.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from kairos.data.schema import make_series, month_start
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.sources.metr import read_curated_csv
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = ["REGULATION", "REGULATORY_LAG", "regulatory_lag"]

OECD_AI: Final = "https://oecd.ai/en/dashboards/overview"

REGULATORY_LAG: Final = SeriesMeta(
    series_id="h.regulatory_lag",
    name="Months from capability to first binding rule",
    unit="months",
    source="oecd_ai",
    source_group="oecd_ai",
    side="H",
    transform="log",
    license="OECD terms and conditions",
    url=OECD_AI,
)


def regulatory_lag(table: pd.DataFrame) -> pd.DataFrame:
    """Compute lags (months) per capability, dated at the rule's month.

    Args:
        table: Columns ``capability``, ``capability_date``, ``rule``, ``rule_date``.

    Returns:
        Columns ``ds``, ``lag_months``, ``capability`` (one row per rule month;
        same-month rules are averaged and their capabilities joined with ``;``).
    """
    cap = month_start(table["capability_date"])
    rule = month_start(table["rule_date"])
    lag = (rule.dt.year - cap.dt.year) * 12 + (rule.dt.month - cap.dt.month)
    if (lag < 0).any():
        raise ValueError("a rule predates its capability")
    df = pd.DataFrame(
        {
            "ds": rule,
            "lag_months": lag.clip(lower=1).astype(float),
            "capability": table["capability"],
        }
    )
    return (
        df.groupby("ds", as_index=False)
        .agg(lag_months=("lag_months", "mean"), capability=("capability", ";".join))
        .sort_values("ds")
        .reset_index(drop=True)
    )


def _build(ctx: BuildContext) -> BuildResult:
    table = read_curated_csv(ctx.files["regulatory_lag.csv"])
    lag = regulatory_lag(table)
    frame = make_series(
        REGULATORY_LAG, lag["ds"], lag["lag_months"], vintage=ctx.vintage,
        extra={"capability": lag["capability"].tolist()},
    )  # fmt: skip
    extra = {"direction": -1, "lag_floor_months": 1, "curated": True}
    return BuildResult(
        datasets=[Dataset(REGULATORY_LAG, frame, extra)], tables={"regulatory_lag": table}
    )


REGULATION: Final = register(
    Source(
        name="regulation",
        title="OECD.AI policy database: capability-to-rule lags",
        license="OECD terms and conditions",
        url=OECD_AI,
        files=(RawFile("regulatory_lag.csv"),),
        build=_build,
    )
)
