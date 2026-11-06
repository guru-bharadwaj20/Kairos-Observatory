import pytest
from hypothesis import given
from hypothesis import strategies as st

from kairos.tripwires.state import DialState, DialStateMachine, Hysteresis

scores = st.lists(st.floats(0.0, 1.0), max_size=60)


def test_enter_and_exit():
    m = DialStateMachine(Hysteresis(enter=0.5, exit=0.3))
    assert m.run([0.4, 0.5, 0.45, 0.31, 0.29]) == [
        DialState.CALM,
        DialState.DANGER,
        DialState.DANGER,
        DialState.DANGER,
        DialState.CALM,
    ]


def test_dwell_delays_switch():
    m = DialStateMachine(Hysteresis(enter=0.5, exit=0.5, dwell=2))
    assert m.run([0.9, 0.1, 0.9, 0.9]) == [
        DialState.CALM,
        DialState.CALM,
        DialState.CALM,
        DialState.DANGER,
    ]


def test_around_keeps_alarm_threshold():
    h = Hysteresis.around(0.2, 0.05)
    assert h.enter == 0.2
    assert h.exit == pytest.approx(0.15)


@pytest.mark.parametrize(
    "kwargs", [{"enter": 0.1, "exit": 0.2}, {"enter": 0.5, "exit": 0.1, "dwell": 0}]
)
def test_invalid_band(kwargs):
    with pytest.raises(ValueError, match="must be"):
        Hysteresis(**kwargs)


def test_nan_rejected():
    with pytest.raises(ValueError, match="finite"):
        DialStateMachine(Hysteresis(0.5, 0.4)).step(float("nan"))


@given(scores, st.floats(0.05, 0.5))
def test_no_flicker_inside_band(xs, width):
    """Scores strictly inside the band never change the state."""
    h = Hysteresis(enter=0.6, exit=0.6 - width)
    for start in DialState:
        m = DialStateMachine(h, state=start)
        inside = [h.exit + (h.enter - h.exit) * x * 0.999 for x in xs]
        assert set(m.run(inside)) <= {start}


@given(scores)
def test_state_reflects_band_edges(xs):
    """With dwell 1, DANGER after a reading >= enter and CALM after one < exit."""
    h = Hysteresis(enter=0.6, exit=0.4)
    m = DialStateMachine(h)
    for x in xs:
        m = m.step(x)
        if x >= h.enter:
            assert m.state is DialState.DANGER
        if x < h.exit:
            assert m.state is DialState.CALM


@given(scores, st.integers(1, 5))
def test_switch_count_bounded_by_dwell(xs, dwell):
    """Each switch needs `dwell` readings, so switches <= len // dwell."""
    states = DialStateMachine(Hysteresis(0.5, 0.5, dwell)).run(xs)
    prev = [DialState.CALM, *states][:-1]
    switches = sum(a is not b for a, b in zip(prev, states, strict=True))
    assert switches <= len(xs) // dwell
