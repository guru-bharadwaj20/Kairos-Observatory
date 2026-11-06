"""Jonas Gap G(t) = ln C - ln H and its velocity dG/dt.

The level of ``G`` is relative to a base year (design note D8); ``dG/dt`` is the
quantity that carries meaning.
"""

from kairos.jonas.gap import (
    GAP_COLUMNS,
    GapSide,
    component_log_index,
    jonas_gap,
    side_components,
    side_index,
    split_sides,
)
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
    "BAND_COVERAGE",
    "GAP_COLUMNS",
    "VELOCITY_COLUMNS",
    "GapSide",
    "HacTest",
    "SustainedResult",
    "component_log_index",
    "decimal_years",
    "gap_slope",
    "gap_velocity",
    "hac_mean_test",
    "jonas_gap",
    "leave_one_out",
    "newey_west_lags",
    "positive_run_length",
    "side_components",
    "side_index",
    "split_sides",
    "sustained_positive",
]
