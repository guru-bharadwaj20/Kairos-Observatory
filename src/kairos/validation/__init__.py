"""Historical validation on runaway curves with known endings.

Public API:

* :class:`Detector` -- the protocol every detector satisfies
  (``detector(frame, as_of) -> Prediction``), and :class:`Prediction`.
* :class:`HyperbolicDetector`, :class:`LPPLSDetector` -- self-contained baselines.
* :func:`run_backtest` / :func:`as_of_view` -- no-lookahead replay.
* :func:`run_case` / :func:`list_cases` -- the registered historical cases.
"""

from kairos.validation.backtest import as_of_view, predictions_frame, run_backtest, yearly_dates
from kairos.validation.cases import CaseResult, list_cases, run_case
from kairos.validation.detectors import (
    DETECTORS,
    Detector,
    HyperbolicDetector,
    Prediction,
    decimal_year,
    from_decimal_year,
    get_detector,
)
from kairos.validation.lppls import LPPLSDetector

__all__ = [
    "DETECTORS",
    "CaseResult",
    "Detector",
    "HyperbolicDetector",
    "LPPLSDetector",
    "Prediction",
    "as_of_view",
    "decimal_year",
    "from_decimal_year",
    "get_detector",
    "list_cases",
    "predictions_frame",
    "run_backtest",
    "run_case",
    "yearly_dates",
]
