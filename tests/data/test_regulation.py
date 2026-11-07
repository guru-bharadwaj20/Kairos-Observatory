import pandas as pd
import pytest

from kairos.data.sources.regulation import regulatory_lag


def table(**over):
    base = {
        "capability": ["a", "b"],
        "capability_date": ["2020-01-15", "2021-03-01"],
        "rule": ["r1", "r2"],
        "rule_date": ["2021-01-02", "2021-03-30"],
    }
    return pd.DataFrame({**base, **over})


def test_lag_months():
    lag = regulatory_lag(table())
    assert lag["lag_months"].tolist() == [12.0, 1.0]  # same-month rule floored at 1


def test_rule_before_capability():
    with pytest.raises(ValueError, match="predates"):
        regulatory_lag(table(rule_date=["2019-01-01", "2021-04-01"]))
