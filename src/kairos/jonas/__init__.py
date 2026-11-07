"""Jonas Gap G(t) = ln C - ln H and its velocity dG/dt.

The level of ``G`` is relative to a base year (design note D8); ``dG/dt`` is the
quantity that carries meaning.
"""

from kairos.jonas.attribution import (
    ATTRIBUTION_COLUMNS,
    GrowthRate,
    component_attribution,
    growth_rate,
)
from kairos.jonas.dial import (
    DIAL_NAME,
    STATES,
    JonasParams,
    JonasState,
    classify,
    jonas_dial,
    load_params,
)
from kairos.jonas.gap import (
    GAP_COLUMNS,
    GapSide,
    component_log_index,
    jonas_gap,
    side_components,
    side_index,
    split_sides,
)
from kairos.jonas.scenarios import SCENARIO_COLUMNS, ClosingScenario, closing_scenarios, to_pct
from kairos.jonas.sustained import (
    HacTest,
    SustainedResult,
    hac_mean_test,
    newey_west_lags,
    positive_run_length,
    sustained_positive,
)
from kairos.jonas.velocity import (
    BAND_COVERAGE,
    VELOCITY_COLUMNS,
    decimal_years,
    gap_slope,
    gap_velocity,
    leave_one_out,
)

__all__ = [
    "ATTRIBUTION_COLUMNS",
    "BAND_COVERAGE",
    "DIAL_NAME",
    "GAP_COLUMNS",
    "SCENARIO_COLUMNS",
    "STATES",
    "VELOCITY_COLUMNS",
    "ClosingScenario",
    "GapSide",
    "GrowthRate",
    "HacTest",
    "JonasParams",
    "JonasState",
    "SustainedResult",
    "classify",
    "closing_scenarios",
    "component_attribution",
    "component_log_index",
    "decimal_years",
    "gap_slope",
    "gap_velocity",
    "growth_rate",
    "hac_mean_test",
    "jonas_dial",
    "jonas_gap",
    "leave_one_out",
    "load_params",
    "newey_west_lags",
    "positive_run_length",
    "side_components",
    "side_index",
    "split_sides",
    "sustained_positive",
    "to_pct",
]
