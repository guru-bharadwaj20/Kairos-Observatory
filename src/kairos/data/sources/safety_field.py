"""H: size of the AI safety field (``h.safety_researchers``, ``h.safety_funding``).

Annual, order-of-magnitude estimates with wide uncertainty bands in
``y_lo``/``y_hi``. Headcount counts full-time-equivalent technical and governance
researchers; funding is philanthropic money for AI safety in USD millions per
year (lab-internal safety budgets are not public and are excluded).
"""

from __future__ import annotations

from typing import Final

from kairos.data.sources.curated import curated_source
from kairos.types import SeriesMeta

__all__ = ["SAFETY_FIELD", "SAFETY_FUNDING", "SAFETY_RESEARCHERS"]

_URL: Final = (
    "https://www.lesswrong.com/posts/7uJnA3XDpTgemRH2c/ai-safety-field-growth-analysis-2025"
)
_LICENSE: Final = "Estimates cited as facts"

SAFETY_RESEARCHERS: Final = SeriesMeta(
    "h.safety_researchers", "AI safety researchers (FTE)", "FTE", "safety_field",
    "field_surveys", "H", "log", _LICENSE, _URL,
)  # fmt: skip
SAFETY_FUNDING: Final = SeriesMeta(
    "h.safety_funding", "Philanthropic AI safety funding", "USD million/yr", "safety_field",
    "grant_databases", "H", "log", _LICENSE,
    "https://www.openphilanthropy.org/focus/potential-risks-advanced-ai/",
)  # fmt: skip

_CAVEAT: Final = {
    "caveats": ["order-of-magnitude estimates", "lab-internal safety work not public"],
    "band": "approx. x0.6 to x1.6-1.7 of the central estimate",
}

SAFETY_FIELD: Final = curated_source(
    name="safety_field",
    title="AI safety field size (headcount and funding estimates)",
    file="safety_field.csv",
    metas={
        SAFETY_RESEARCHERS.series_id: SAFETY_RESEARCHERS,
        SAFETY_FUNDING.series_id: SAFETY_FUNDING,
    },
    extra={SAFETY_RESEARCHERS.series_id: _CAVEAT, SAFETY_FUNDING.series_id: _CAVEAT},
)
