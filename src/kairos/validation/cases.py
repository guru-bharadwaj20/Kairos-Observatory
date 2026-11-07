"""Registry of historical validation cases.

A case is a module-level function that takes no required arguments and returns
a :class:`CaseResult`. Cases are registered by dotted path so listing them does
not import their dependencies.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

__all__ = ["CASES", "CaseResult", "list_cases", "run_case"]


@dataclass(frozen=True)
class CaseResult:
    """Outcome of one validation case.

    Attributes:
        case: Case name.
        passed: Whether the acceptance criterion was met. Failures are kept and
            reported, never hidden.
        summary: Headline numbers (JSON-serialisable scalars).
        criterion: The acceptance criterion in words.
        table: Per-as-of-date results, if the case is a replay.
        notes: Caveats a reader needs to interpret the result.
        skipped: The case could not run (e.g. data unavailable offline).
    """

    case: str
    passed: bool
    summary: dict[str, Any]
    criterion: str
    table: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)
    skipped: bool = False


CASES: dict[str, str] = {
    "population": "kairos.validation.population:run",
    "dotcom": "kairos.validation.bubbles:run_dotcom",
    "bitcoin": "kairos.validation.bubbles:run_bitcoin",
    "covid": "kairos.validation.covid:run",
}
"""``case name -> "module:function"``."""


def list_cases() -> list[str]:
    """Names of the registered cases, sorted."""
    return sorted(CASES)


def run_case(name: str, **kwargs: Any) -> CaseResult:
    """Import and run a registered case."""
    try:
        target = CASES[name]
    except KeyError:
        raise KeyError(f"unknown case {name!r}; known: {list_cases()}") from None
    module_name, func_name = target.split(":")
    func = cast(Callable[..., CaseResult], getattr(importlib.import_module(module_name), func_name))
    return func(**kwargs)
