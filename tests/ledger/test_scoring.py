from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from kairos.ledger.chain import Ledger
from kairos.ledger.conditional import EventAssumption
from kairos.ledger.scoring import (
    ForecastScore,
    crps_quantiles,
    interventions_since,
    log_intervention,
    pit_quantiles,
    score_matured,
)

T0 = datetime(2026, 11, 6, 12, tzinfo=UTC)
MANIFEST = "a" * 64


def actuals_for(forecast, factor=1.0, series_id=None, drop_last=False):
    points = forecast.points[:-1] if drop_last else forecast.points
    return pd.DataFrame(
        {
            "ds": pd.to_datetime([p.ds for p in points]),
            "y": [p.at(0.5) * factor for p in points],
            "series_id": series_id or forecast.series_id,
        }
    )


def test_crps_approximates_gaussian_closed_form():
    levels = np.linspace(0.005, 0.995, 199)
    values = stats.norm.ppf(levels)
    y = 0.7
    exact = y * (2 * stats.norm.cdf(y) - 1) + 2 * stats.norm.pdf(y) - 1 / np.sqrt(np.pi)
    assert crps_quantiles(levels, values, y) == pytest.approx(exact, rel=0.02)


def test_crps_zero_for_perfect_point_forecast():
    assert crps_quantiles([0.5], [3.0], 3.0) == 0.0
    with pytest.raises(ValueError, match="same length"):
        crps_quantiles([0.5], [], 1.0)


def test_pit_interpolates_and_clamps():
    levels, values = [0.05, 0.5, 0.95], [1.0, 2.0, 3.0]
    assert pit_quantiles(levels, values, 2.0) == pytest.approx(0.5)
    assert pit_quantiles(levels, values, 2.5) == pytest.approx(0.725)
    assert pit_quantiles(levels, values, -10) == pytest.approx(0.05)
    assert pit_quantiles(levels, values, 99) == pytest.approx(0.95)
    assert 0 <= pit_quantiles(levels, [1.0, 1.0, 1.0], 1.0) <= 1


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "forecasts.jsonl")


def test_not_scored_before_maturity(ledger, make_forecast):
    f = make_forecast()
    ledger.append_forecast(f, manifest_hash=MANIFEST, recorded_at=T0)
    assert score_matured(ledger, actuals_for(f), today=date(2027, 3, 31)) == []
    # matured by date but last actual not yet published
    assert score_matured(ledger, actuals_for(f, drop_last=True), today=date(2027, 6, 1)) == []
    assert all(e.kind == "forecast" for e in ledger.entries())


def test_matured_forecast_scored_once_and_appended(ledger, make_forecast):
    f = make_forecast().with_claims([140.0])
    fe = ledger.append_forecast(f, manifest_hash=MANIFEST, recorded_at=T0)
    (score,) = score_matured(ledger, actuals_for(f), today=date(2027, 4, 1))
    assert score.forecast_entry == fe.entry_hash
    assert score.coverage == 1.0
    assert score.mean_log_gap == pytest.approx(0.0, abs=1e-12)
    assert score.counterfactual is False
    assert score.claims[0].crossed == date(2027, 2, 1)
    assert score.claims[0].within_interval is True
    last = ledger.head()
    assert last.kind == "score"
    assert ForecastScore.model_validate(last.payload).forecast_hash == fe.payload["forecast_hash"]
    assert score_matured(ledger, actuals_for(f), today=date(2027, 5, 1)) == []


def test_miss_lowers_coverage(ledger, make_forecast):
    f = make_forecast()
    ledger.append_forecast(f, manifest_hash=MANIFEST, recorded_at=T0)
    (score,) = score_matured(ledger, actuals_for(f, factor=2.0), today=date(2027, 4, 1))
    assert score.coverage == 0.0
    assert score.mean_crps > 0
    assert all(p.pit == pytest.approx(0.95) for p in score.points)
    assert score.mean_log_gap == pytest.approx(np.log(2.0))


def test_counterfactual_when_intervention_logged_after(ledger, make_forecast):
    assumed = EventAssumption(name="old_rule", ds=date(2025, 1, 1))
    f = make_forecast(events=(assumed,))
    ledger.append_forecast(f, manifest_hash=MANIFEST, recorded_at=T0)
    later = timedelta(days=30)
    hit = log_intervention(
        ledger,
        EventAssumption(name="compute_cap", ds=date(2027, 1, 1)),
        series_ids=["frontier_compute"],
        recorded_at=T0 + later,
    )
    log_intervention(  # other series: ignored
        ledger, EventAssumption(name="x", ds=date(2027, 1, 1)), series_ids=["other"]
    )
    log_intervention(  # after the horizon: ignored
        ledger, EventAssumption(name="y", ds=date(2028, 1, 1))
    )
    log_intervention(  # already assumed: ignored
        ledger, EventAssumption(name="old_rule", ds=date(2025, 1, 1))
    )
    entries = ledger.entries()
    assert [e.entry_hash for e in interventions_since(entries, entries[0])] == [hit.entry_hash]
    (score,) = score_matured(ledger, actuals_for(f, factor=0.7), today=date(2027, 4, 1))
    assert score.counterfactual is True
    assert score.interventions == (hit.entry_hash,)
    assert score.mean_log_gap == pytest.approx(np.log(0.7))


def test_superseded_forecasts_are_not_scored(ledger, make_forecast):
    f = make_forecast()
    fe = ledger.append_forecast(f, manifest_hash=MANIFEST, recorded_at=T0)
    ledger.append_correction(fe.entry_hash, "wrong unit")
    assert score_matured(ledger, actuals_for(f), today=date(2027, 4, 1)) == []


def test_dry_run_does_not_append(ledger, make_forecast):
    f = make_forecast()
    ledger.append_forecast(f, manifest_hash=MANIFEST, recorded_at=T0)
    assert len(score_matured(ledger, actuals_for(f), today=date(2027, 4, 1), append=False)) == 1
    assert len(ledger.entries()) == 1
