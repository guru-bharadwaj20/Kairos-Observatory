import json
from datetime import date

import pytest
from pydantic import ValidationError

from kairos.ledger.conditional import ConditionalClaim, EventAssumption, crossing_claim
from kairos.ledger.forecast import Forecast, forecast_hash

EVENT = EventAssumption(name="eu_ai_act_gpai", ds=date(2025, 8, 2), description="GPAI duties")


def test_crossing_dates_from_quantiles(make_forecast):
    # median = 100 * 1.1**i at months 2026-11 .. 2027-04; q95 = 1.25x, q05 = 0.8x
    f = make_forecast()
    claim = crossing_claim(f, 140.0, assumed_events=(EVENT,))
    assert claim.earliest == date(2026, 12, 1)  # 1.25 * 121 >= 140
    assert claim.by == date(2027, 2, 1)  # 146.4
    assert claim.latest == date(2027, 4, 1)  # 0.8 * 177.2 = 141.7
    assert claim.coverage == 0.9
    text = claim.statement()
    assert text == (
        "If no new intervention, frontier_compute reaches 140 FLOP by 2027-02-01 "
        "(90% interval: 2026-12-01 to 2027-04-01). Events assumed: eu_ai_act_gpai."
    )


def test_open_ended_interval(make_forecast):
    claim = crossing_claim(make_forecast(), 160.0)
    assert claim.by == date(2027, 3, 1)
    assert claim.latest is None
    assert "to after 2027-04-01" in claim.statement()
    assert claim.resolved_on() == date(2027, 4, 1)


def test_never_crosses(make_forecast):
    claim = crossing_claim(make_forecast(), 1e6)
    assert (claim.earliest, claim.by, claim.latest) == (None, None, None)
    assert "does not reach" in claim.statement()
    assert "Events assumed: none" in claim.statement()


def test_missing_quantile_level(make_forecast):
    with pytest.raises(KeyError):
        crossing_claim(make_forecast(), 140.0, coverage=0.8)


def test_inconsistent_claim_rejected():
    with pytest.raises(ValidationError, match="earliest <= by <= latest"):
        ConditionalClaim(
            series_id="x",
            threshold=1.0,
            by=date(2027, 1, 1),
            earliest=date(2027, 6, 1),
            latest=None,
            horizon_end=date(2027, 12, 1),
            coverage=0.9,
        )


def test_forecast_stores_claims_and_events(make_forecast):
    f = make_forecast(events=(EVENT,)).with_claims([140.0, 160.0])
    assert len(f.claims) == 2
    assert f.claims[0].assumed_events == (EVENT,)
    payload = json.loads(json.dumps(f.payload()))
    assert payload["claims"][0]["condition"] == "no new intervention"
    assert payload["events"][0]["name"] == "eu_ai_act_gpai"
    restored = Forecast.model_validate(payload)
    assert restored == f
    assert forecast_hash(payload) == forecast_hash(f)


def test_unconditional_hash_unchanged_by_new_optional_fields(make_forecast):
    f = make_forecast()
    assert "claims" not in f.payload()
    assert "events" not in f.payload()


def test_future_events_cannot_be_assumed(make_forecast):
    future = EventAssumption(name="later", ds=date(2027, 1, 1))
    with pytest.raises(ValidationError, match="after as_of"):
        make_forecast(events=(future,))


def test_claim_must_match_series_and_events(make_forecast):
    f = make_forecast()
    claim = crossing_claim(f, 140.0)
    with pytest.raises(ValidationError, match="not 'frontier_compute'"):
        make_forecast(claims=(claim.model_copy(update={"series_id": "other"}),))
    with pytest.raises(ValidationError, match="assumed_events"):
        make_forecast(claims=(claim,), events=(EVENT,))
