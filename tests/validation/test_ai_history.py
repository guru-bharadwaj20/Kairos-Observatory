from datetime import date

import numpy as np
import pytest

from kairos.validation import run_case
from kairos.validation.ai_history import frontier_frame, load_models, segmented_changepoints


def test_frontier_has_no_lookahead():
    models = load_models()
    frame = frontier_frame(models, date(2016, 1, 1))
    assert frame["ds"].max() < np.datetime64("2016-01-02")
    assert frame["y"].is_monotonic_increasing
    later = models[models["publication_date"] >= "2016-01-02"]
    assert frame["y"].max() < later["training_compute_flop"].max()


def test_changepoint_recovers_known_break():
    t = np.arange(2010, 2020, 1 / 12)
    ly = np.where(t < 2015, 0.5 * (t - 2010), 2.5 + 2.0 * (t - 2015))
    cps = segmented_changepoints(t, np.exp(ly), max_breaks=1)
    assert cps.breaks[0] == pytest.approx(2015, abs=0.3)
    assert cps.slopes == pytest.approx((0.5, 2.0), abs=0.05)


def test_replay_covers_every_year():
    result = run_case("ai_history", last_year=2020)
    assert result.passed
    assert result.summary["years"] == 9
    assert list(result.table["as_of"]) == [date(y, 1, 1) for y in range(2012, 2021)]
