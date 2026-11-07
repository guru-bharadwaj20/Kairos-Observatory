import pandas as pd

from kairos.data.sources.base import BuildContext, fetch_source
from kairos.data.sources.validation import POPULATION_TABLE, VALIDATION_GROWTH


def build(src, tmp_path):
    f = fetch_source(src, tmp_path, offline=True)
    return src.build(BuildContext(files=f.files, vintage=f.vintage))


def test_growth_cases(tmp_path):
    res = build(VALIDATION_GROWTH, tmp_path)
    by_id = {d.meta.series_id: d.frame for d in res.datasets}
    pop = by_id["val.population"]
    assert pop["ds"].min() == pd.Timestamp("1700-01-01")
    assert 2.4e9 < pop.loc[pop["ds"] == "1950-01-01", "y"].iloc[0] < 2.7e9
    tr = by_id["val.transistors"]
    assert tr["y"].iloc[-1] > 1e6 * tr["y"].iloc[0]
    hist = res.tables[POPULATION_TABLE]
    assert hist["year"].min() < 0  # BCE years kept in the table
    assert (res.datasets[0].frame["side"] == "validation").all()
