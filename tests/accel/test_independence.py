from __future__ import annotations

import pandas as pd
import pytest

from kairos.accel.independence import confirm_by_source_group, source_groups


def _sig(ids, flags):
    return pd.DataFrame({"series_id": ids, "significant": flags})


def test_three_groups_confirm():
    sig = _sig(["a", "b", "c", "d"], [True, True, True, False])
    groups = {"a": "epoch", "b": "metr", "c": "openalex", "d": "epoch"}
    conf = confirm_by_source_group(sig, groups)
    assert conf.confirmed
    assert conf.n_groups == 3


def test_same_source_does_not_count_twice():
    sig = _sig(["a", "b", "c"], [True, True, True])
    conf = confirm_by_source_group(sig, {"a": "epoch", "b": "epoch", "c": "metr"})
    assert not conf.confirmed
    assert conf.n_groups == 2
    assert conf.groups["epoch"] == ["a", "b"]


def test_groups_from_series_frame():
    frame = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01", "2020-01-01"]),
            "y": [1.0, 2.0, 3.0, 4.0],
            "series_id": ["a", "a", "b", "c"],
            "source_group": ["g1", "g1", "g2", None],
        }
    )
    assert source_groups(frame) == {"a": "g1", "b": "g2", "c": "c"}
    conf = confirm_by_source_group(_sig(["a", "b", "c"], [True] * 3), frame)
    assert conf.confirmed


def test_errors():
    frame = pd.DataFrame({"series_id": ["a", "a"], "source_group": ["g1", "g2"]})
    with pytest.raises(ValueError, match="more than one"):
        source_groups(frame)
    with pytest.raises(KeyError, match="zz"):
        confirm_by_source_group(_sig(["zz"], [True]), {})
    with pytest.raises(ValueError, match="min_groups"):
        confirm_by_source_group(_sig([], []), {}, min_groups=0)
