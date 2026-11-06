"""Epoch AI "Data on Notable AI Models" loader.

Produces the per-model table ``epoch_models`` (one row per model with a known
publication date): ``model``, ``organization``, ``ds`` (publication day),
``parameters``, ``training_compute_flop``, ``confidence`` and ``frontier``
(Epoch's "Frontier model" flag). Monthly series built from it live in
:mod:`kairos.data.frontier` (K-2.06) and :mod:`kairos.data.sources.drivers` (K-2.11).

Licence: CC BY 4.0, attribution "Epoch AI".
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from kairos.data.sources.base import BuildContext, BuildResult, RawFile, Source, register

__all__ = ["EPOCH", "EPOCH_LICENSE", "EPOCH_URL", "MODELS_TABLE", "parse_models"]

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


def _build(ctx: BuildContext) -> BuildResult:
    models = parse_models(ctx.files["notable_ai_models.csv"])
    return BuildResult(tables={MODELS_TABLE: models})


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
