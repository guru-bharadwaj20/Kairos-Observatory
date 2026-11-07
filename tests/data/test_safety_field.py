from kairos.data.sources.base import BuildContext, fetch_source
from kairos.data.sources.safety_field import SAFETY_FIELD


def test_two_series_with_bands(tmp_path):
    f = fetch_source(SAFETY_FIELD, tmp_path, offline=True)
    res = SAFETY_FIELD.build(BuildContext(files=f.files, vintage=f.vintage))
    by_id = {d.meta.series_id: d for d in res.datasets}
    assert set(by_id) == {"h.safety_researchers", "h.safety_funding"}
    for d in by_id.values():
        fr = d.frame
        assert (fr["y_lo"] < fr["y"]).all()
        assert (fr["y_hi"] > fr["y"]).all()
        assert (fr["side"] == "H").all()
