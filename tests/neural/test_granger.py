from __future__ import annotations

import pytest

from kairos.neural import synthetic_series
from kairos.neural.granger import (
    GrangerResult,
    agreement_table,
    self_lag_f_test,
    var_granger,
)
from kairos.neural.kappa import KappaResult
from kairos.neural.regressors import design_frame

DRIVERS = {"log_compute": 6, "log_investment": 12, "log_researchers": 12}


def _design(coupling):
    return design_frame(synthetic_series(192, coupling=coupling, seed=4, noise=0.05), "capability")


def test_self_lag_test_finds_coupling():
    strong = self_lag_f_test(_design(0.9), DRIVERS, self_lags=6, difference=False)
    assert strong.p_value < 0.01
    assert strong.df_num == 6
    none = self_lag_f_test(_design(0.0), DRIVERS, self_lags=6, difference=False)
    assert none.p_value > strong.p_value


def test_self_lag_test_differenced_runs():
    res = self_lag_f_test(_design(0.5), DRIVERS, self_lags=3)
    assert 0.0 <= res.p_value <= 1.0
    assert res.n_obs > 100


def test_self_lag_test_too_short():
    short = design_frame(synthetic_series(20, seed=0), "capability")
    with pytest.raises(ValueError, match="too few"):
        self_lag_f_test(short, DRIVERS, self_lags=6)


def test_var_granger_drivers_cause_target():
    res = var_granger(_design(0.0), list(DRIVERS), maxlags=12, difference=False)
    assert res.p_value < 0.05
    assert res.lags >= 1


def test_agreement_table():
    def kap(p):
        return KappaResult(0.2, 0.1, 0.3, 0.9, p, (), (0.2,), 1.0, 0.8)

    def gr(p):
        return GrangerResult(3.0, p, 6, 100, 120, 6)

    table = agreement_table({"a": (kap(0.01), gr(0.001)), "b": (kap(0.01), gr(0.4))})
    assert list(table["agree"]) == [True, False]
    assert list(table["kappa_sig"]) == [True, True]
