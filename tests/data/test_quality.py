import dataclasses

import pytest

from kairos.data.quality import (
    QualityError,
    check_dataset,
    outlier_flags,
    quality_report,
    raise_on_errors,
)
from kairos.data.store import Dataset


def checks(d):
    return {i.check for i in check_dataset(d)}


def test_clean(make_dataset):
    assert checks(make_dataset()) == set()


def test_outlier_flagged(make_dataset):
    d = make_dataset(n=36)
    d.frame.loc[20, "y"] *= 1e4
    assert outlier_flags(d).iloc[20]
    assert "outliers" in checks(d)


def test_gap_and_monotonic(make_dataset):
    d = make_dataset(n=24)
    gap = dataclasses.replace(d, frame=d.frame.drop(index=[5, 6]).reset_index(drop=True))
    assert "gaps" in checks(gap)
    rev = Dataset(d.meta, d.frame.iloc[::-1].reset_index(drop=True))
    assert "monotonic" in checks(rev)


def test_unit_and_side(make_dataset):
    d = make_dataset()
    f = d.frame.copy()
    f.loc[0, "unit"] = "W"
    f["side"] = "H"
    assert {"unit", "side"} <= checks(Dataset(d.meta, f))


def test_report_raises(make_dataset):
    d = make_dataset()
    f = d.frame.copy()
    f.loc[0, "unit"] = "W"
    rep = quality_report([d, Dataset(d.meta, f)])
    assert len(rep) == 1
    with pytest.raises(QualityError, match="unit"):
        raise_on_errors(rep)
    raise_on_errors(quality_report([d]))
