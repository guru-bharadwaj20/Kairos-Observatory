"""K-9.06: Moore's law false-alarm test.

Transistor counts grew exponentially for fifty years: fast, but with a constant
doubling time, so there is no finite-time singularity. A detector that alarms
on this series is confusing "fast" with "accelerating". Replayed every year
from 1985 to 2021, the acceleration detectors must stay at **Level 0**.

Data: Our World in Data, transistors per microprocessor (Karl Rupp / Wikipedia),
CC BY 4.0. Years with no new record repeat the previous maximum in the source.
"""

from __future__ import annotations

from kairos.types import SeriesFrame, SeriesMeta
from kairos.validation.backtest import predictions_frame, run_backtest, yearly_dates
from kairos.validation.cases import CaseResult
from kairos.validation.datasets import SNAPSHOTS, load_snapshot
from kairos.validation.detectors import Detector, HyperbolicDetector

__all__ = ["load_transistors", "run"]

SERIES_ID = "transistors_per_chip"

SNAPSHOTS[SERIES_ID] = (
    "transistors.csv",
    SeriesMeta(
        series_id=SERIES_ID,
        name="Transistors per microprocessor",
        unit="transistors",
        source="owid",
        source_group="semiconductors",
        side="validation",
        transform="log",
        license="CC-BY-4.0",
        url="https://ourworldindata.org/grapher/transistors-per-microprocessor",
    ),
)


def load_transistors() -> SeriesFrame:
    """Transistor-count snapshot, 1971-2021."""
    return load_snapshot(SERIES_ID)


def run(*, detector: Detector | None = None, n_jobs: int | None = 1) -> CaseResult:
    """Replay ``detector`` (default hyperbolic baseline) yearly 1985-2021."""
    detector = detector or HyperbolicDetector()
    table = predictions_frame(
        run_backtest(load_transistors(), detector, yearly_dates(1985, 2021), n_jobs=n_jobs)
    )
    max_level = int(table["level"].max())
    return CaseResult(
        case="moore",
        passed=max_level == 0,
        criterion="level 0 at every yearly as-of date 1985-2021",
        summary={
            "as_of_dates": len(table),
            "max_level": max_level,
            "alarms": int(table["alarm"].sum()),
            "false_alarm_rate": round(float(table["alarm"].mean()), 3),
        },
        table=table,
    )
