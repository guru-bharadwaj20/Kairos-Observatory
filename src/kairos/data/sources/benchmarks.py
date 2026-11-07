"""Benchmark state-of-the-art histories (``c.bench.<benchmark>``).

For each benchmark the SOTA series keeps only **record months**: months where the
best reported score rises above every earlier score. Scores are fractions in
(0, 1) and the series carry ``transform="logit"``, so models fit them on the logit
scale and saturation near 100% does not read as a slowdown.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from kairos.data.schema import make_series, month_start
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.sources.metr import read_curated_csv
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = ["BENCHMARKS", "BENCHMARK_NAMES", "SCORES_TABLE", "benchmark_meta", "sota_history"]

SCORES_TABLE: Final = "benchmark_scores"
BENCHMARK_NAMES: Final = {
    "mmlu": "MMLU",
    "gpqa_diamond": "GPQA Diamond",
    "swe_bench_verified": "SWE-bench Verified",
}
_LICENSE: Final = "Facts from public model cards and papers"


def benchmark_meta(benchmark: str) -> SeriesMeta:
    """Return the metadata of the SOTA series of one benchmark."""
    return SeriesMeta(
        series_id=f"c.bench.{benchmark}",
        name=f"{BENCHMARK_NAMES.get(benchmark, benchmark)} state of the art",
        unit="fraction",
        source="benchmarks",
        source_group="developer_reports",
        side="C",
        transform="logit",
        license=_LICENSE,
        url="https://paperswithcode.com/",
    )


def sota_history(scores: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-model scores of one benchmark to its record months.

    Args:
        scores: Columns ``release_date``, ``score`` and ``model``.

    Returns:
        Columns ``ds`` (month start), ``score``, ``model`` for each new record.
    """
    df = scores.assign(ds=month_start(scores["release_date"]).to_numpy())
    df = df.sort_values(["ds", "score"]).groupby("ds", as_index=False).last()
    best = df["score"].cummax().shift(fill_value=-1.0)
    return df.loc[df["score"] > best, ["ds", "score", "model"]].reset_index(drop=True)


def _build(ctx: BuildContext) -> BuildResult:
    scores = read_curated_csv(ctx.files["benchmark_scores.csv"])
    out: list[Dataset] = []
    for bench, grp in scores.groupby("benchmark", sort=True):
        hist = sota_history(grp)
        meta = benchmark_meta(str(bench))
        frame = make_series(
            meta,
            hist["ds"],
            hist["score"],
            vintage=str(ctx.vintage),
            extra={"model": hist["model"].tolist()},
        )
        out.append(Dataset(meta, frame, {"reduction": "record months", "curated": True}))
    return BuildResult(datasets=out, tables={SCORES_TABLE: scores})


BENCHMARKS: Final = register(
    Source(
        name="benchmarks",
        title="Benchmark SOTA histories (model cards, papers)",
        license=_LICENSE,
        url="https://paperswithcode.com/",
        files=(RawFile("benchmark_scores.csv"),),
        build=_build,
    )
)
