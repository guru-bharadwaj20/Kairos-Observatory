"""Event calendar schema and conversion to Prophet/NeuralProphet holiday frames.

An event frame has one row per dated event:

================= ============== ==================================================
column            dtype          meaning
================= ============== ==================================================
``event``         str            short snake_case name, e.g. ``gpt4_release``
``ds``            datetime64[ns] day of the event (not snapped to month start)
``type``          str            one of :data:`EVENT_TYPES`
``lower_window``  int64          days before ``ds`` the effect may start (``<= 0``)
``upper_window``  int64          days after ``ds`` the effect may last (``>= 0``)
``source_url``    str            at least one ``http(s)`` URL backing the event
``scope``         str            ``ai`` or a validation case such as ``covid``
``description``   str            one line of context
================= ============== ==================================================
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pandas as pd
import pandera.pandas as pa

__all__ = [
    "EVENT_COLUMNS",
    "EVENT_SCHEMA",
    "EVENT_TYPES",
    "read_event_csv",
    "to_holidays",
    "validate_events",
]

EVENT_TYPES: Final = ("release", "regulation", "export_control", "incident", "intervention")
"""``intervention`` covers non-AI validation events such as COVID lockdowns."""

EVENT_COLUMNS: Final = (
    "event",
    "ds",
    "type",
    "lower_window",
    "upper_window",
    "source_url",
    "scope",
    "description",
)

EVENT_SCHEMA: Final = pa.DataFrameSchema(
    {
        "event": pa.Column(str, pa.Check.str_matches(r"^[a-z0-9][a-z0-9_]*$")),
        "ds": pa.Column("datetime64[ns]"),
        "type": pa.Column(str, pa.Check.isin(EVENT_TYPES)),
        "lower_window": pa.Column("int64", pa.Check.le(0)),
        "upper_window": pa.Column("int64", pa.Check.ge(0)),
        "source_url": pa.Column(str, pa.Check.str_matches(r"^https?://\S+")),
        "scope": pa.Column(str, pa.Check.str_length(min_value=1)),
        "description": pa.Column(str),
    },
    unique=["event", "ds"],
    strict=True,
    ordered=False,
    name="KairosEvents",
)
"""Schema of the event calendar."""


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an event frame and return it sorted by ``ds``.

    Args:
        df: Frame with the :data:`EVENT_COLUMNS`.

    Raises:
        pandera.errors.SchemaErrors: If any row breaks the schema.
    """
    out = EVENT_SCHEMA.validate(df.reset_index(drop=True), lazy=True)
    return out.sort_values(["ds", "event"], kind="stable").reset_index(drop=True)


def read_event_csv(path: Path) -> pd.DataFrame:
    """Read a curated event CSV (``#`` lines are comments) and validate it.

    Missing ``scope`` defaults to ``ai`` and missing ``description`` to ``""``.
    """
    df = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    df["ds"] = pd.to_datetime(df["ds"]).astype("datetime64[ns]")
    for col in ("lower_window", "upper_window"):
        df[col] = df[col].astype("int64")
    if "scope" not in df:
        df["scope"] = "ai"
    if "description" not in df:
        df["description"] = ""
    return validate_events(df[list(EVENT_COLUMNS)])


def to_holidays(
    events: pd.DataFrame,
    *,
    types: Iterable[str] | None = None,
    scope: str | None = "ai",
) -> pd.DataFrame:
    """Convert events to the ``holidays`` frame Prophet and NeuralProphet expect.

    Args:
        events: A validated event frame.
        types: Keep only these event types (all when ``None``).
        scope: Keep only this scope (all when ``None``).

    Returns:
        Columns ``holiday``, ``ds``, ``lower_window``, ``upper_window``.
    """
    keep = pd.Series(True, index=events.index)
    if types is not None:
        keep &= events["type"].isin(list(types))
    if scope is not None:
        keep &= events["scope"] == scope
    sel = events.loc[keep]
    return pd.DataFrame(
        {
            "holiday": sel["event"].to_numpy(),
            "ds": sel["ds"].to_numpy(),
            "lower_window": sel["lower_window"].to_numpy(),
            "upper_window": sel["upper_window"].to_numpy(),
        }
    )
