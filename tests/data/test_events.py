import pandas as pd
import pandera.errors as pe
import pytest

from kairos.data.events import read_event_csv, to_holidays, validate_events

CSV = """# comment line
event,ds,type,lower_window,upper_window,source_url,scope,description
gpt4_release,2023-03-14,release,0,30,https://openai.com/research/gpt-4,ai,GPT-4
uk_lockdown,2020-03-23,intervention,0,60,https://www.gov.uk/x,covid,Lockdown
eu_ai_act,2024-08-01,regulation,-30,0,https://eur-lex.europa.eu/x,ai,AI Act
"""


@pytest.fixture
def events(tmp_path):
    p = tmp_path / "events.csv"
    p.write_text(CSV)
    return read_event_csv(p)


def test_read_sorted(events):
    assert events["event"].tolist() == ["uk_lockdown", "gpt4_release", "eu_ai_act"]
    assert events["ds"].dtype == "datetime64[ns]"


def test_holidays_filters(events):
    hol = to_holidays(events)
    assert list(hol.columns) == ["holiday", "ds", "lower_window", "upper_window"]
    assert set(hol["holiday"]) == {"gpt4_release", "eu_ai_act"}
    assert to_holidays(events, types=["release"])["holiday"].tolist() == ["gpt4_release"]
    assert len(to_holidays(events, scope=None)) == 3


@pytest.mark.parametrize(
    ("col", "value"),
    [("type", "party"), ("lower_window", 3), ("source_url", "not a url"), ("event", "Bad Name")],
)
def test_invalid(events, col, value):
    df = events.copy()
    df.loc[0, col] = value
    with pytest.raises(pe.SchemaErrors):
        validate_events(df)


def test_missing_url_rejected(events):
    df = events.copy()
    df.loc[0, "source_url"] = ""
    with pytest.raises(pe.SchemaErrors):
        validate_events(df)


def test_scope_default(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text(
        "event,ds,type,lower_window,upper_window,source_url\na,2020-01-01,incident,0,0,http://x.y\n"
    )
    df = read_event_csv(p)
    assert df.loc[0, "scope"] == "ai"
    assert df.loc[0, "ds"] == pd.Timestamp("2020-01-01")
