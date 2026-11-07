"""AI share of AI research: a documented proxy (``c.ai_research_share``).

**What is measured.** The estimated fraction of sentences in new arXiv computer
science abstracts that were substantially written or modified by a large
language model, from the distributional estimator of Liang et al. (2024),
"Mapping the Increasing Use of LLMs in Scientific Papers", arXiv:2404.01268.

**Why it is a proxy.** The quantity KAIROS wants is the share of AI research
*work* done by AI systems (the recursion in the idea's §9c). No series measures
that directly. LLM-modified writing is a lower-bound signal of AI involvement in
the research pipeline that is measured the same way over time.

**Caveats.**

* It measures writing help, not ideas, experiments or code.
* Points are read from the paper's figures and rounded; bounds are wide on purpose.
* Detection-based estimates may drift as models change style and as authors edit.
* Coverage ends at the paper's last month; refreshes need newer estimates.
"""

from __future__ import annotations

from typing import Final

from kairos.data.sources.curated import curated_source
from kairos.types import SeriesMeta

__all__ = ["AI_RESEARCH_SHARE", "CAVEATS", "RESEARCH_SHARE"]

AI_RESEARCH_SHARE: Final = SeriesMeta(
    series_id="c.ai_research_share",
    name="Share of LLM-modified sentences in arXiv CS abstracts (proxy)",
    unit="fraction",
    source="research_share",
    source_group="liang2024",
    side="C",
    transform="logit",
    license="arXiv paper; figures cited as facts",
    url="https://arxiv.org/abs/2404.01268",
)

CAVEATS: Final = (
    "writing assistance only, not research labour",
    "values read from published figures and rounded",
    "detector drift as model style changes",
)

RESEARCH_SHARE: Final = curated_source(
    name="research_share",
    title="Liang et al. (2024), LLM-modified content in arXiv CS abstracts",
    file="ai_research_share.csv",
    metas={AI_RESEARCH_SHARE.series_id: AI_RESEARCH_SHARE},
    extra={
        AI_RESEARCH_SHARE.series_id: {
            "proxy": True,
            "method": "distributional LLM-usage estimator (Liang et al. 2024)",
            "caveats": list(CAVEATS),
        }
    },
)
