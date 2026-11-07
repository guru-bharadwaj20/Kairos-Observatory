"""Event calendar v1: frontier releases 2012-2026, major laws, export controls, incidents.

The curated CSV follows :data:`kairos.data.events.EVENT_SCHEMA`; every event has at
least one source URL. Use :func:`kairos.data.events.to_holidays` to feed it to
Prophet or NeuralProphet.
"""

from __future__ import annotations

from typing import Final

from kairos.data.events import read_event_csv
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register

__all__ = ["CALENDAR", "CALENDAR_VERSION"]

CALENDAR_VERSION: Final = "v1"


def _build(ctx: BuildContext) -> BuildResult:
    return BuildResult(events=read_event_csv(ctx.files["events.csv"]))


CALENDAR: Final = register(
    Source(
        name="calendar",
        title=f"KAIROS event calendar {CALENDAR_VERSION}",
        license="CC-BY-4.0 (KAIROS compilation; facts cited per row)",
        url="https://github.com/guru-bharadwaj20/Kairos-Observatory",
        files=(RawFile("events.csv"),),
        build=_build,
    )
)
