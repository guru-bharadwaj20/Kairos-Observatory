import pandas as pd
import pytest

from kairos.data.sources.literacy import interpolate_surveys


def test_band_widens_between_surveys():
    pts = pd.DataFrame(
        {"date": ["2020-01-01", "2021-01-01"], "value": [0.2, 0.4], "se": [0.01, 0.01]}
    )
    m = interpolate_surveys(pts)
    assert len(m) == 13
    assert m["y"].iloc[6] == pytest.approx(0.3)
    assert m["sd"].iloc[0] == pytest.approx(0.01)
    assert m["sd"].iloc[-1] == pytest.approx(0.01)
    assert m["sd"].iloc[6] == m["sd"].max()
    assert m["sd"].iloc[3] < m["sd"].iloc[6]
    assert (m["y_lo"] <= m["y"]).all()
    assert (m["y_hi"] >= m["y"]).all()
    assert m["interpolated"].sum() == 11
