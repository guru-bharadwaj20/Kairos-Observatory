"""KAIROS Observatory: an open early-warning system for runaway AI.

KAIROS watches AI capability ``C(t)`` against human oversight capacity ``H(t)``
and reports four dials (acceleration, recursion, Vinge Horizon, Jonas Gap)
that feed pre-agreed tripwire levels. See ``KAIROS_Project_Idea.md``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kairos-observatory")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
