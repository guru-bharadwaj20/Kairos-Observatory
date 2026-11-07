import pandas as pd

from kairos.data.events import EVENT_TYPES, to_holidays
from kairos.data.sources.base import BuildContext, fetch_source
from kairos.data.sources.calendar import CALENDAR


def test_calendar_v1(tmp_path):
    f = fetch_source(CALENDAR, tmp_path, offline=True)
    ev = CALENDAR.build(BuildContext(files=f.files, vintage=f.vintage)).events
    assert ev is not None
    assert ev["ds"].min() < pd.Timestamp("2013-01-01")
    assert ev["ds"].max() >= pd.Timestamp("2025-01-01")
    assert {"release", "regulation", "export_control", "incident"} <= set(ev["type"])
    assert set(ev["type"]) <= set(EVENT_TYPES)
    assert ev["source_url"].str.startswith("http").all()
    assert len(to_holidays(ev)) == len(ev)
