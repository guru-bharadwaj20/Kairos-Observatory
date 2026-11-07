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


def test_bubble_cases(tmp_path):
    from kairos.data.sources.validation import VALIDATION_BUBBLES

    res = build(VALIDATION_BUBBLES, tmp_path)
    by_id = {d.meta.series_id: d.frame for d in res.datasets}
    nq = by_id["val.nasdaq"]
    assert len(nq) == 84
    peak = nq.loc[nq["y"].idxmax(), "ds"]
    assert pd.Timestamp("1999-12-01") <= peak <= pd.Timestamp("2000-03-01")
    btc = by_id["val.bitcoin"]
    assert btc.loc[btc["y"].idxmax(), "ds"] == pd.Timestamp("2017-12-01")
    assert by_id["val.covid_cases"]["ds"].max() == pd.Timestamp("2022-12-01")
    assert res.events is not None
    assert (res.events["scope"] == "covid").all()


def test_monthly_mean_drops_missing():
    from kairos.data.sources.validation import monthly_mean

    m = monthly_mean(
        pd.Series(["2020-01-01", "2020-01-02", "2020-02-01"]), pd.Series(["1", ".", "3"])
    )
    assert m["y"].tolist() == [1.0, 3.0]
