"""Lagged exogenous drivers: compute, investment and researchers.

Each driver is a :class:`DriverSpec`: the ``SeriesFrame`` series it comes from,
the column name it gets in the NeuralProphet frame (``log_compute``, ...), the
transform applied (``log`` by default) and its **own lag window** ``n_lags``.
Investment and talent act on capability with longer delays than compute, so
they get longer windows.

:func:`design_frame` joins the target and the drivers into one wide frame on
common months; :func:`add_lagged_drivers` registers the drivers on a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from kairos.neural.model import to_np_frame
from kairos.types import SeriesFrame, Transform

if TYPE_CHECKING:
    from neuralprophet import NeuralProphet

__all__ = ["DEFAULT_DRIVERS", "DriverSpec", "add_lagged_drivers", "design_frame"]


@dataclass(frozen=True, slots=True)
class DriverSpec:
    """One lagged exogenous driver.

    Attributes:
        column: Column name in the NeuralProphet frame.
        series_id: ``SeriesFrame.series_id`` of the driver (K-2.11).
        n_lags: Lag window in months fed to the model.
        transform: Transform applied to the raw values.
        regularization: Optional sparsity penalty on the driver's weights.
    """

    column: str
    series_id: str
    n_lags: int
    transform: Transform = "log"
    regularization: float | None = None

    def __post_init__(self) -> None:
        if self.n_lags < 1:
            raise ValueError(f"{self.column}: n_lags must be >= 1")


DEFAULT_DRIVERS: tuple[DriverSpec, ...] = (
    DriverSpec("log_compute", "compute", n_lags=6),
    DriverSpec("log_investment", "investment", n_lags=12),
    DriverSpec("log_researchers", "researchers", n_lags=12),
)
"""Drivers used for ``κ`` (roadmap task K-2.11): one lag window per driver."""


def design_frame(
    frame: SeriesFrame,
    target_id: str,
    drivers: Sequence[DriverSpec] = DEFAULT_DRIVERS,
    *,
    target_transform: Transform = "log",
) -> pd.DataFrame:
    """Build the wide NeuralProphet frame ``ds, y, <driver columns>``.

    Args:
        frame: Long ``SeriesFrame`` holding the target and every driver.
        target_id: Series ID of the target.
        drivers: Driver specs; their columns appear in this order.
        target_transform: Transform of the target.

    Returns:
        Months present in every series, sorted, with no missing values.

    Raises:
        ValueError: If column names collide or no month is shared.
    """
    columns = [d.column for d in drivers]
    if len(set(columns)) != len(columns) or "y" in columns or "ds" in columns:
        raise ValueError(f"driver columns must be unique and not 'ds'/'y': {columns}")
    out = to_np_frame(frame, target_id, transform=target_transform)
    for spec in drivers:
        part = to_np_frame(frame, spec.series_id, transform=spec.transform, column=spec.column)
        out = out.merge(part, on="ds", how="inner")
    out = out.dropna().reset_index(drop=True)
    if out.empty:
        raise ValueError("target and drivers share no months")
    return out


def add_lagged_drivers(model: NeuralProphet, drivers: Sequence[DriverSpec]) -> NeuralProphet:
    """Register every driver as a lagged regressor with its own window."""
    for spec in drivers:
        model.add_lagged_regressor(
            spec.column, n_lags=spec.n_lags, regularization=spec.regularization
        )
    return model
