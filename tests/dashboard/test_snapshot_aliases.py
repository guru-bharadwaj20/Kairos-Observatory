"""Dial-module names are accepted by the dashboard snapshot parser."""

from kairos.dashboard.snapshot import DIAL_ALIASES, DIAL_ORDER


def test_aliases_map_to_dashboard_dials() -> None:
    """Every alias targets a known dashboard dial."""
    assert DIAL_ALIASES == {"vinge": "vinge_horizon", "jonas": "jonas_gap"}
    assert set(DIAL_ALIASES.values()) <= set(DIAL_ORDER)
