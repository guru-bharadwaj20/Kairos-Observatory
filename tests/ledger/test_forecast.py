import hashlib
import json
from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from kairos.ledger.canonical import canonical_float, canonical_json, is_sha256_hex, sha256_hex
from kairos.ledger.forecast import Forecast, ForecastPoint, forecast_hash, quantile_key
from kairos.types import DialReading


def test_canonical_sorts_keys_and_strips_whitespace():
    assert canonical_json({"b": 1, "a": [1.0, None, True]}) == b'{"a":[1.0,null,true],"b":1}'


def test_canonical_is_ascii_and_escapes_unicode():
    assert canonical_json({"δ": "κ"}) == b'{"\\u03b4":"\\u03ba"}'


@pytest.mark.parametrize(
    ("x", "text"),
    [(1.0, "1.0"), (-0.0, "0.0"), (0.1 + 0.2, "0.3"), (1e-7, "1e-07"), (2.5e20, "2.5e+20")],
)
def test_canonical_float_format(x, text):
    assert canonical_float(x) == text


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rejected(bad):
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": bad})


def test_int_and_float_are_distinct():
    assert canonical_json(1) != canonical_json(1.0)


def test_datetimes_normalised_to_utc():
    ist = timezone(timedelta(hours=5, minutes=30))
    a = datetime(2026, 1, 1, 5, 30, tzinfo=ist)
    b = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    assert canonical_json(a) == canonical_json(b) == b'"2026-01-01T00:00:00.000000Z"'
    with pytest.raises(ValueError, match="naive"):
        canonical_json(datetime(2026, 1, 1))


def test_non_string_keys_rejected():
    with pytest.raises(TypeError):
        canonical_json({1: "a"})


def test_sha256_matches_hashlib():
    assert sha256_hex({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()
    assert is_sha256_hex(sha256_hex(b"x"))
    assert not is_sha256_hex("ABC")


@given(
    st.recursive(
        st.none()
        | st.booleans()
        | st.integers()
        | st.floats(allow_nan=False, allow_infinity=False)
        | st.text(),
        lambda c: st.lists(c) | st.dictionaries(st.text(), c),
        max_leaves=20,
    )
)
def test_canonical_is_valid_json_and_stable(value):
    first = canonical_json(value)
    assert canonical_json(json.loads(first)) == first


def test_same_forecast_same_hash(make_forecast):
    assert forecast_hash(make_forecast()) == forecast_hash(make_forecast())
    assert forecast_hash(make_forecast()) != forecast_hash(make_forecast(base=101.0))


def test_hash_survives_payload_round_trip(make_forecast):
    f = make_forecast()
    restored = json.loads(json.dumps(f.payload()))
    assert forecast_hash(restored) == forecast_hash(f)


def test_hash_ignores_last_ulp_noise(make_forecast):
    f = make_forecast()
    p = f.points[0]
    noisy = ForecastPoint(ds=p.ds, quantiles={k: v * (1 + 1e-16) for k, v in p.quantiles.items()})
    g = f.model_copy(update={"points": (noisy, *f.points[1:])})
    assert forecast_hash(g) == forecast_hash(f)


def test_hash_with_dials(make_forecast):
    reading = DialReading(
        dial="vinge", value=12.0, level=1, ci_lo=8, ci_hi=15, as_of=date(2026, 10, 1)
    )
    f = make_forecast().model_copy(update={"dials": (reading,)})
    assert forecast_hash(f) != forecast_hash(make_forecast())
    assert forecast_hash(json.loads(json.dumps(f.payload()))) == forecast_hash(f)


def test_quantile_keys_canonical():
    assert quantile_key(0.05) == "0.05"
    assert quantile_key(0.5) == "0.5"
    p = ForecastPoint(ds=date(2027, 1, 1), quantiles={"0.050": 1.0, 0.95: 2.0})
    assert set(p.quantiles) == {"0.05", "0.95"}
    assert p.at(0.95) == 2.0


def test_validation_rules(make_forecast):
    with pytest.raises(ValidationError, match="non-decreasing"):
        ForecastPoint(ds=date(2027, 1, 1), quantiles={0.05: 2.0, 0.95: 1.0})
    f = make_forecast()
    with pytest.raises(ValidationError, match="after as_of"):
        Forecast.model_validate({**f.payload(), "as_of": "2030-01-01"})
    with pytest.raises(ValidationError, match="timezone-aware"):
        Forecast.model_validate({**f.payload(), "issued_at": "2026-11-06T12:00:00"})
    with pytest.raises(ValidationError, match="strictly increasing"):
        Forecast.model_validate({**f.payload(), "points": [f.payload()["points"][0]] * 2})
