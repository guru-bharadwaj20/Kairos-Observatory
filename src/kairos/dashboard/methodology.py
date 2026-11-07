"""Plain-language methodology text, built from the snapshot's published parameters.

The numbers (λ, the alarm threshold, the tolerance) come from the snapshot, so
the page always describes the parameters the published results used.
"""

from __future__ import annotations

from kairos.dashboard.snapshot import Snapshot
from kairos.dashboard.theme import DIALS, LEVELS

#: The design notes D1-D12 from docs/design.md, rewritten for a general reader.
DESIGN_NOTES: tuple[tuple[str, str, str], ...] = (
    ("D1", "Measuring acceleration",
     "Growth rates are estimated from historical data with a smoothing method, never from "
     "a model's own forecast, which would hide any acceleration by construction."),
    ("D2", "Measuring forecast error",
     "Errors are measured as a factor (for example, \"within 2x\"), not as a percentage of "
     "a logarithm, which would be meaningless."),
    ("D3", "Defining the Vinge Horizon",
     "h* is the first horizon where the smoothed error leaves the tolerance, so one lucky "
     "long-range forecast cannot inflate it."),
    ("D4", "Honest uncertainty for δ",
     "Intervals for δ come from a block bootstrap, because the data are serially correlated "
     "and textbook standard errors would be far too narrow."),
    ("D5", "Time remaining",
     "The formula for time to blow-up, 1/(δ·g), was checked algebraically and is reported "
     "with an interval."),
    ("D6", "The Jonas rule",
     "An alarm is raised when the probability of danger exceeds 1/(1+λ). This was checked "
     "and is implemented exactly as written."),
    ("D7", "Building the frontier series",
     "Model releases are irregular, so a dedicated method builds the capability frontier "
     "and records how it did so."),
    ("D8", "What the Jonas Gap level means",
     "C and H are in different units, so the gap's level is arbitrary. Only its change "
     "over time, dG/dt, is used as a signal."),
    ("D9", "Avoiding overfitting in κ",
     "κ is measured only out of sample, across several seeds, and compared with a "
     "shuffled-data baseline so that noise is not mistaken for signal."),
    ("D10", "Many metrics, few false alarms",
     "When many metrics are tested, a false-discovery correction is applied, and metrics "
     "from the same source do not count as independent confirmation."),
    ("D11", "Proving when forecasts were made",
     "Forecasts go into a hash-chained ledger anchored with OpenTimestamps, because git "
     "history alone can be rewritten."),
    ("D12", "Toolchain",
     "The project pins Python 3.12 so the forecasting libraries install reliably."),
)  # fmt: skip

ASSUMPTIONS: tuple[str, ...] = (
    'Prophet and NeuralProphet describe "normal" progress; departures from them are the '
    "signal, not the forecast.",
    "Every series is used only as it was known at each cutoff date (no lookahead).",
    "Probabilities of danger come from bootstrap and ensemble distributions, not point estimates.",
    "Decisions are also checked in the worst plausible scenario (Rawls maximin).",
    "No dial may depend only on data controlled by a single measured lab (Löb).",
)


def alarm_threshold(lam: float) -> float:
    """Jonas rule: alarm when ``P(danger) > 1 / (1 + λ)`` (D6)."""
    if lam <= 0:
        raise ValueError("λ must be positive")
    return 1.0 / (1.0 + lam)


def methodology_markdown(snap: Snapshot) -> str:
    """The methodology page as Markdown."""
    p = snap.parameters
    thr = alarm_threshold(p.lam)
    out = [
        "## The one number philosophy supplies: λ",
        f"KAIROS treats a missed catastrophe as **λ = {p.lam:g}** times worse than a false "
        f"alarm. A dial therefore raises its alarm once the probability of danger exceeds "
        f"**1/(1+λ) = {thr:.1%}**. λ and every threshold are published in versioned YAML "
        f"(tripwires v{p.tripwires_version}) and set through public deliberation, never "
        f"hidden in code.",
        "## The four dials",
        "| Dial | What it measures | Danger sign |",
        "|---|---|---|",
        *(f"| {d.title} | {d.measures} | {d.danger_sign} |" for d in DIALS.values()),
        "## Tripwire levels",
        "| Level | Trigger | Recommended response |",
        "|---|---|---|",
        *(f"| {lv.number}: {lv.name} | {lv.trigger} | {lv.response} |" for lv in LEVELS),
        "KAIROS recommends; humans decide. It is an instrument, not an automatic switch.",
        "## Vinge Horizon tolerance",
        f"h* is computed for forecasts that stay within a factor of "
        f"**{p.vinge_tolerance_factor:g}x** of what actually happened.",
        "## Assumptions",
        *(f"- {a}" for a in ASSUMPTIONS),
        "## Design notes D1-D12",
        "| Note | Topic | In plain language |",
        "|---|---|---|",
        *(f"| {n} | {topic} | {text} |" for n, topic, text in DESIGN_NOTES),
    ]
    return "\n\n".join(_join_tables(out))


def _join_tables(lines: list[str]) -> list[str]:
    """Keep table rows on consecutive lines (Markdown needs no blank lines there)."""
    blocks: list[str] = []
    for line in lines:
        if line.startswith("|") and blocks and blocks[-1].startswith("|"):
            blocks[-1] += "\n" + line
        else:
            blocks.append(line)
    return blocks
