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

__all__ = [
    "GAP_COLUMNS",
    "GapSide",
    "component_log_index",
    "jonas_gap",
    "side_components",
    "side_index",
    "split_sides",
]
