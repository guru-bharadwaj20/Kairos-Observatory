import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from kairos.philosophy.jonas import JonasPolicy, alarm_threshold, load_policy


def test_threshold_formula():
    assert alarm_threshold(9.0) == pytest.approx(0.1)
    assert alarm_threshold(1.0) == pytest.approx(0.5)


@given(st.floats(0.01, 1000), st.floats(0.0, 1.0))
def test_alarm_matches_loss_comparison(lam, p):
    pol = JonasPolicy(lam=lam)
    losses = pol.expected_losses(p)
    assert pol.alarm(p) == (losses["silent"] > losses["alarm"])


@given(st.floats(0.01, 1000))
def test_alarm_flips_at_threshold(lam):
    pol = JonasPolicy(lam=lam)
    t = pol.threshold
    assert not pol.alarm(max(0.0, t - 1e-6))
    assert pol.alarm(min(1.0, t + 1e-6))


def test_invalid_inputs():
    with pytest.raises(ValidationError):
        JonasPolicy(lam=0)
    with pytest.raises(ValueError, match="must be in"):
        JonasPolicy(lam=1).alarm(1.5)


def test_packaged_policy_loads():
    spec = load_policy()
    assert spec.jonas.lam > 0
    assert spec.version
    h = spec.jonas.hysteresis()
    assert h.enter == pytest.approx(spec.jonas.threshold)
    assert h.exit <= h.enter


def test_policy_from_path(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text('version: "x"\njonas:\n  lambda: 3\n', encoding="utf-8")
    assert load_policy(p).jonas.threshold == pytest.approx(0.25)
