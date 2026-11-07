"""``kairos governor``: turn dial readings into a tripwire level."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from kairos.types import DialReading

app = typer.Typer(
    help="Ethical governor: dial readings -> tripwire level 0-4.", no_args_is_help=True
)

SpecOpt = Annotated[Path | None, typer.Option(help="tripwires.yaml (default: packaged).")]
PolicyOpt = Annotated[Path | None, typer.Option(help="policy.yaml (default: packaged).")]


def read_readings(path: Path) -> list[DialReading]:
    """Read a JSON array of :class:`DialReading` objects."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise typer.BadParameter("readings file must hold a JSON array")
    return [DialReading.model_validate(r) for r in raw]


@app.command()
def evaluate(
    readings: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="JSON array of DialReading.")
    ],
    spec: SpecOpt = None,
    policy: PolicyOpt = None,
    history: Annotated[bool, typer.Option(help="Print one alert per as_of date.")] = False,
    allow_normal_fallback: Annotated[
        bool, typer.Option(help="Approximate P(danger) from the interval when a dial has no draws.")
    ] = False,
) -> None:
    """Evaluate readings week by week (grouped by ``as_of``) and print the alert JSON."""
    from kairos.philosophy.jonas import load_policy
    from kairos.tripwires.aggregator import Governor, load_tripwire_spec

    gov = Governor(
        load_tripwire_spec(spec), load_policy(policy), allow_normal_fallback=allow_normal_fallback
    )
    weeks: dict[date, list[DialReading]] = defaultdict(list)
    for r in read_readings(readings):
        weeks[r.as_of].append(r)
    if not weeks:
        raise typer.BadParameter("no readings")
    alerts = [gov.evaluate(weeks[d]).model_dump(mode="json") for d in sorted(weeks)]
    typer.echo(json.dumps(alerts if history else alerts[-1], indent=2))


@app.command("spec")
def show_spec(spec: SpecOpt = None, policy: PolicyOpt = None) -> None:
    """Print the tripwire levels and the Jonas threshold."""
    from kairos.philosophy.jonas import load_policy
    from kairos.tripwires.aggregator import load_tripwire_spec

    s = load_tripwire_spec(spec)
    pol = load_policy(policy).jonas
    typer.echo(
        f"tripwires {s.version}; lambda={pol.lam:g}, alarm when P(danger) > {pol.threshold:.3f}"
    )
    for lv in s.levels:
        typer.echo(f"{lv.level} {lv.name}: {lv.trigger} -> {lv.response}")
