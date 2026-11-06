from __future__ import annotations

import numpy as np
import pytest

from kairos.jonas import ATTRIBUTION_COLUMNS, component_attribution, growth_rate, jonas_gap


def test_growth_rate_exact(frame_factory) -> None:
    gap = jonas_gap(frame_factory(), 2016).set_index("ds")
    g = growth_rate(gap["ln_h"], 36)
    assert g.rate == pytest.approx(0.25, rel=1e-9)
    assert g.n == 36
    with pytest.raises(ValueError, match="3 observations"):
        growth_rate(gap["ln_h"].iloc[:2])


def test_attribution_ranks_slowest_first(frame_factory) -> None:
    frame = frame_factory(h_rates={"h_papers": 0.4, "h_funding": 0.1, "h_evals": 0.25}, noise=0.0)
    att = component_attribution(frame, 2016)
    assert tuple(att.columns) == ATTRIBUTION_COLUMNS
    assert att["series_id"].tolist() == ["h_funding", "h_evals", "h_papers"]
    assert att["rank"].tolist() == [1, 2, 3]
    # g_C = 0.8; lags are 0.7, 0.55, 0.4
    np.testing.assert_allclose(att["lag"], [0.7, 0.55, 0.4], rtol=1e-9)
    # contributions sum to dG/dt = 0.8 - 0.25
    assert att["contribution"].sum() == pytest.approx(0.55, rel=1e-9)
    # dropping the laggard narrows the gap the most
    assert att.loc[0, "loo_dgdt"] == att["loo_dgdt"].min()


def test_attribution_weights(frame_factory) -> None:
    att = component_attribution(frame_factory(), 2016, weights={"h_papers": 3.0})
    w = dict(zip(att["series_id"], att["weight"], strict=True))
    assert w == pytest.approx({"h_papers": 0.75, "h_funding": 0.25})
    assert att["contribution"].sum() == pytest.approx(0.8 - (0.75 * 0.3 + 0.25 * 0.2))


def test_single_h_component_has_no_loo(frame_factory) -> None:
    att = component_attribution(frame_factory(h_rates={"h": 0.3}), 2016)
    assert len(att) == 1
    assert np.isnan(att.loc[0, "loo_dgdt"])
