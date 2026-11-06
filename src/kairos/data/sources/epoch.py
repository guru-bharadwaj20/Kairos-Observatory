"""Epoch AI "Data on Notable AI Models" loader.

Produces the per-model table ``epoch_models`` (one row per model with a known
publication date): ``model``, ``organization``, ``ds`` (publication day),
``parameters``, ``training_compute_flop``, ``confidence`` and ``frontier``
(Epoch's "Frontier model" flag). Monthly series built from it live in
:mod:`kairos.data.frontier` (K-2.06) and :mod:`kairos.data.sources.drivers` (K-2.11).

Licence: CC BY 4.0, attribution "Epoch AI".
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd

from kairos.data.frontier import EnvelopeSpec, frontier_envelope
from kairos.data.schema import make_series
from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register
from kairos.data.store import Dataset
from kairos.types import SeriesMeta

__all__ = [
    "EPOCH",
    "EPOCH_LICENSE",
    "EPOCH_URL",
    "FRONTIER_COMPUTE",
    "FRONTIER_SPEC",
    "MODELS_TABLE",
    "frontier_compute",
    "parse_models",
]

EPOCH_URL: Final = "https://epoch.ai/data/notable-ai-models"
EPOCH_CSV: Final = "https://epoch.ai/data/notable_ai_models.csv"
EPOCH_LICENSE: Final = "CC-BY-4.0"
MODELS_TABLE: Final = "epoch_models"

_COLUMNS: Final = {
    "Model": "model",
    "Organization": "organization",
    "Publication date": "ds",
    "Parameters": "parameters",
    "Training compute (FLOP)": "training_compute_flop",
    "Confidence": "confidence",
    "Frontier model": "frontier",
}


def parse_models(path: Path) -> pd.DataFrame:
    """Parse the Epoch notable-models CSV into the ``epoch_models`` table.

    Rows without a parseable publication date are dropped. Columns missing from the
    file (Epoch occasionally renames them) are filled with nulls.

    Args:
        path: Local copy of ``notable_ai_models.csv``.

    Returns:
        The model table sorted by ``ds`` then ``model``.
    """
    raw = pd.read_csv(path, low_memory=False)
    df = pd.DataFrame({new: raw.get(old, pd.NA) for old, new in _COLUMNS.items()})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").astype("datetime64[ns]")
    df = df.dropna(subset=["ds", "model"])
    for col in ("parameters", "training_compute_flop"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["frontier"] = df["frontier"].astype(str).str.lower().isin({"true", "1", "yes"})
    for col in ("model", "organization", "confidence"):
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df.sort_values(["ds", "model"], kind="stable").reset_index(drop=True)


FRONTIER_SPEC: Final = EnvelopeSpec(method="top_k", k=3, interpolation="loglinear")
"""Envelope used for ``c.frontier_compute``: log-mean of the 3 largest runs so far."""

FRONTIER_COMPUTE: Final = SeriesMeta(
    series_id="c.frontier_compute",
    name="Frontier training compute (top-3 envelope of notable models)",
    unit="FLOP",
    source="epoch",
    source_group="epoch",
    side="C",
    transform="log",
    license=EPOCH_LICENSE,
    url=EPOCH_URL,
)


def frontier_compute(
    models: pd.DataFrame, vintage: date, spec: EnvelopeSpec = FRONTIER_SPEC
) -> Dataset:
    """Build the monthly frontier-compute series from the ``epoch_models`` table.

    Args:
        models: Output of :func:`parse_models`.
        vintage: Retrieval date of the Epoch data.
        spec: Envelope parameters (stored in the dataset's extra metadata).
    """
    env = frontier_envelope(
        models.rename(columns={"training_compute_flop": "value"}), spec, value="value"
    )
    extra_cols = ["record", "interpolated", "gap", "months_since_record"]
    frame = make_series(
        FRONTIER_COMPUTE,
        env["ds"],
        env["y"],
        vintage=vintage,
        extra={c: env[c].tolist() for c in extra_cols},
    )
    return Dataset(FRONTIER_COMPUTE, frame, {"envelope": spec.as_metadata()})


def _build(ctx: BuildContext) -> BuildResult:
    models = parse_models(ctx.files["notable_ai_models.csv"])
    return BuildResult(
        datasets=[frontier_compute(models, ctx.vintage)], tables={MODELS_TABLE: models}
    )


EPOCH: Final = register(
    Source(
        name="epoch",
        title="Epoch AI, Data on Notable AI Models",
        license=EPOCH_LICENSE,
        url=EPOCH_URL,
        files=(RawFile("notable_ai_models.csv", EPOCH_CSV),),
        build=_build,
    )
)
