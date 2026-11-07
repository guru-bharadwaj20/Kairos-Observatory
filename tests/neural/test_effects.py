from __future__ import annotations

import numpy as np
import pytest

from kairos.neural import NeuralConfig, synthetic_series
from kairos.neural.effects import event_effect_table
from kairos.neural.events import EventSpec
from kairos.neural.regressors import design_frame

CFG = NeuralConfig(
    epochs=150, batch_size=16, learning_rate=0.1, n_changepoints=0, early_stopping=False
)


def test_effect_table_recovers_shock():
    # A one-month shock of +0.5 log units on 2014-01 and nothing on 2016-01.
    sf = synthetic_series(96, seed=0, coupling=0.0, drivers={}, noise=0.01)
    bump = (sf.series_id == "capability") & (sf.ds == "2014-01-01")
    sf.loc[bump, "y"] *= np.exp(0.5)
    design = design_frame(sf, "capability", drivers=())
    events = [EventSpec.of("shock", ["2014-01-01"]), EventSpec.of("dud", ["2016-01-01"])]
    table = event_effect_table(design, events, config=CFG, n_seeds=3, n_jobs=1).set_index("event")
    assert table.loc["shock", "n_seeds"] == 3
    assert table.loc["shock", "effect"] > 0.25
    assert table.loc["shock", "factor"] == pytest.approx(np.exp(table.loc["shock", "effect"]))
    assert abs(table.loc["dud", "effect"]) < table.loc["shock", "effect"]
    assert table.loc["shock", "ci_lo"] <= table.loc["shock", "ci_hi"]


def test_effect_table_needs_events():
    design = design_frame(synthetic_series(24, seed=0, drivers={}), "capability", drivers=())
    with pytest.raises(ValueError, match="no events"):
        event_effect_table(design, [], n_seeds=1)
