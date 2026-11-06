import numpy as np
import pandas as pd
import pytest

from kairos.data.schema import make_series
from kairos.data.store import Dataset
from kairos.types import SeriesMeta


def meta(series_id="c.test", side="C", transform="log", unit="FLOP"):
    return SeriesMeta(
        series_id=series_id,
        name=series_id,
        unit=unit,
        source="unit",
        source_group="unit",
        side=side,
        transform=transform,
        license="CC0-1.0",
        url="https://example.org",
    )


def dataset(series_id="c.test", side="C", n=24, start="2020-01-01", growth=0.05, **kw):
    ds = pd.date_range(start, periods=n, freq="MS")
    y = np.exp(growth * np.arange(n))
    m = meta(series_id, side, **kw)
    return Dataset(m, make_series(m, ds, y, vintage="2026-11-01"), {"note": "synthetic"})


@pytest.fixture
def root(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def make_dataset():
    return dataset


@pytest.fixture
def make_meta():
    return meta
