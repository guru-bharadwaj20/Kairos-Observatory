"""Colours and level labels.

Colours come from the Okabe-Ito palette, which stays distinguishable under the
common forms of colour blindness. Colour is never the only cue: every level
and dial state also carries a text label and a distinct symbol.
"""

from __future__ import annotations

from dataclasses import dataclass

BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#595959"
INK = "#1A1A1A"

#: C(t) and H(t) keep the same colours everywhere.
CAPABILITY_COLOR = VERMILLION
HUMAN_COLOR = BLUE

STATE_STYLE = {
    "calm": (BLUE, "●", "Calm"),
    "danger": (VERMILLION, "▲", "Danger zone"),
}


@dataclass(frozen=True)
class Level:
    """A tripwire level: trigger, response and display style."""

    number: int
    name: str
    trigger: str
    response: str
    background: str
    foreground: str
    symbol: str


#: Defaults mirror the tripwire table in KAIROS_Project_Idea.md §8. The
#: published snapshot may override name and response (tripwires.yaml, K-8.08).
LEVELS: tuple[Level, ...] = (
    Level(0, "Normal", "All dials calm", "Weekly publication", BLUE, "#FFFFFF", "○"),
    Level(
        1, "Watch", "One dial in the danger zone",
        "More data collection, public notice", YELLOW, INK, "◔",
    ),
    Level(
        2, "Warning", "Two dials, or one confirmed on 3 or more independent metrics",
        "Mandatory third-party evaluation before new frontier deployments", ORANGE, INK, "◑",
    ),
    Level(
        3, "Alarm", "Three dials",
        "Pause scaling past the current frontier until H catches up; "
        "emergency funding for the H side", VERMILLION, INK, "◕",
    ),
    Level(
        4, "Kairos", "All four dials",
        "The decisive moment, the last window in which intervention is still cheap",
        PURPLE, INK, "●",
    ),
)  # fmt: skip


def level(n: int) -> Level:
    """Return the spec for alert level ``n`` (0-4)."""
    if not 0 <= n < len(LEVELS):
        raise ValueError(f"alert level must be 0-4, got {n}")
    return LEVELS[n]


@dataclass(frozen=True)
class DialInfo:
    """Plain-language description of a dial."""

    title: str
    measures: str
    danger_sign: str


#: Wording from KAIROS_Project_Idea.md §8.
DIALS: dict[str, DialInfo] = {
    "acceleration": DialInfo(
        "Acceleration", "Whether growth is super-exponential, and when it would blow up",
        "δ significantly above 0 across several independent metrics",
    ),
    "recursion": DialInfo(
        "Recursion", "How much AI progress is driven by AI itself",
        "κ rising across rolling windows",
    ),
    "vinge_horizon": DialInfo(
        "Vinge Horizon", "How far ahead we can still predict", "h* shrinking",
    ),
    "jonas_gap": DialInfo(
        "Jonas Gap", "Capability outrunning comprehension", "dG/dt > 0, sustained",
    ),
}  # fmt: skip
