"""``kairos run weekly``: the weekly KAIROS pipeline orchestrator.

The pipeline runs six stages in a fixed order::

    refresh -> fit -> dials -> governor -> ledger -> export

Each stage lives in another package (data, models, dials, governor, ledger,
dashboard). The orchestrator does not import them at module load time.
Instead every stage lists one or more *entry points* (``module:function``)
that are tried in order. The first one that imports wins. If none can be
imported the stage is reported as ``unavailable`` and the pipeline carries
on, so the command works on a partial checkout and picks stages up as they
land.

A stage entry point has the signature ``(ctx: RunContext) -> Mapping | None``.
What it returns is stored in ``ctx.artifacts[stage_name]`` so later stages
can read it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "WEEKLY_STAGES",
    "RunContext",
    "RunReport",
    "Stage",
    "StageResult",
    "StageStatus",
    "build_parser",
    "main",
    "resolve_entry_point",
    "run_pipeline",
]

log = logging.getLogger("kairos.run")

StageFn = Callable[["RunContext"], Mapping[str, Any] | None]


class StageStatus(StrEnum):
    """Outcome of a single stage."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    SKIPPED = "skipped"
    DRY_RUN = "dry-run"


@dataclass(frozen=True)
class Stage:
    """One pipeline stage and the entry points that may implement it."""

    name: str
    entry_points: tuple[str, ...]
    description: str
    needs_gpu: bool = False


@dataclass
class RunContext:
    """State shared by every stage of one run."""

    run_id: str
    as_of: datetime
    root: Path
    cpu_only: bool = False
    dry_run: bool = False
    artifacts: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def device(self) -> str:
        """Device hint for stages: ``cpu`` when the run is CPU-only."""
        return "cpu" if self.cpu_only else "auto"


@dataclass(frozen=True)
class StageResult:
    """What happened to one stage."""

    name: str
    status: StageStatus
    seconds: float = 0.0
    entry_point: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "name": self.name,
            "status": str(self.status),
            "seconds": round(self.seconds, 3),
            "entry_point": self.entry_point,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RunReport:
    """Summary of a whole run, written as JSON for CI and the ledger."""

    run_id: str
    as_of: str
    results: tuple[StageResult, ...]

    @property
    def failed(self) -> bool:
        """Whether any stage failed."""
        return any(r.status is StageStatus.FAILED for r in self.results)

    @property
    def unavailable(self) -> tuple[str, ...]:
        """Names of stages with no importable entry point."""
        return tuple(r.name for r in self.results if r.status is StageStatus.UNAVAILABLE)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "run_id": self.run_id,
            "as_of": self.as_of,
            "failed": self.failed,
            "stages": [r.to_dict() for r in self.results],
        }


WEEKLY_STAGES: tuple[Stage, ...] = (
    Stage(
        "refresh",
        ("kairos.data.refresh:run_stage", "kairos.data:refresh"),
        "Refresh C(t), H(t) and the event calendar; write the data manifest",
    ),
    Stage(
        "fit",
        ("kairos.models.weekly:run_stage", "kairos.models:fit_all"),
        "Fit Prophet null models and NeuralProphet ensembles",
        needs_gpu=True,
    ),
    Stage(
        "dials",
        ("kairos.dials.weekly:run_stage", "kairos.dials:compute_all"),
        "Compute the four dials: delta/t_c, kappa, h*, dG/dt",
    ),
    Stage(
        "governor",
        ("kairos.governor.weekly:run_stage", "kairos.governor:evaluate"),
        "Apply the ethical governor and set the tripwire level",
    ),
    Stage(
        "ledger",
        ("kairos.ledger.weekly:run_stage", "kairos.ledger:append_weekly"),
        "Hash the forecast, append it to the ledger, timestamp it",
    ),
    Stage(
        "export",
        ("kairos.dashboard.export:run_stage", "kairos.dashboard:export_static"),
        "Export the static dashboard snapshot",
    ),
)


def resolve_entry_point(spec: str) -> StageFn | None:
    """Import ``module:attr`` and return it, or ``None`` if it is missing.

    Only a missing module or attribute counts as "unavailable". An import
    that fails for another reason (a bug in the stage) is re-raised so it is
    reported as a failure rather than hidden.
    """
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if module_name == missing or module_name.startswith(missing + "."):
            return None
        raise
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        return None
    result: StageFn = fn
    return result


def _run_stage(stage: Stage, ctx: RunContext) -> StageResult:
    for spec in stage.entry_points:
        try:
            fn = resolve_entry_point(spec)
        except Exception as exc:
            return StageResult(stage.name, StageStatus.FAILED, 0.0, spec, f"import error: {exc!r}")
        if fn is None:
            continue
        if ctx.dry_run:
            return StageResult(stage.name, StageStatus.DRY_RUN, 0.0, spec, "would run")
        start = time.perf_counter()
        try:
            out = fn(ctx)
        except Exception as exc:
            log.exception("stage %s failed", stage.name, extra={"run_id": ctx.run_id})
            return StageResult(
                stage.name, StageStatus.FAILED, time.perf_counter() - start, spec, repr(exc)
            )
        if out is not None:
            ctx.artifacts[stage.name] = out
        return StageResult(stage.name, StageStatus.OK, time.perf_counter() - start, spec)
    return StageResult(
        stage.name,
        StageStatus.UNAVAILABLE,
        detail="stage unavailable: none of " + ", ".join(stage.entry_points),
    )


def run_pipeline(
    ctx: RunContext,
    stages: Sequence[Stage] = WEEKLY_STAGES,
    *,
    only: Sequence[str] = (),
    skip: Sequence[str] = (),
    keep_going: bool = False,
) -> RunReport:
    """Run ``stages`` in order and return a report.

    A failed stage stops the run (later stages are ``skipped``) unless
    ``keep_going`` is set. Unavailable stages never stop the run.
    """
    known = {s.name for s in stages}
    unknown = sorted((set(only) | set(skip)) - known)
    if unknown:
        raise ValueError(f"unknown stage(s): {', '.join(unknown)}")

    results: list[StageResult] = []
    stopped = False
    for stage in stages:
        if stopped:
            results.append(StageResult(stage.name, StageStatus.SKIPPED, detail="earlier failure"))
            continue
        if (only and stage.name not in only) or stage.name in skip:
            results.append(StageResult(stage.name, StageStatus.SKIPPED, detail="deselected"))
            continue
        log.info("stage %s: %s", stage.name, stage.description, extra={"run_id": ctx.run_id})
        result = _run_stage(stage, ctx)
        log.info("stage %s -> %s", stage.name, result.status, extra={"run_id": ctx.run_id})
        results.append(result)
        if result.status is StageStatus.FAILED and not keep_going:
            stopped = True
    return RunReport(ctx.run_id, ctx.as_of.isoformat(), tuple(results))


def build_parser() -> argparse.ArgumentParser:
    """Build the ``kairos run`` argument parser."""
    parser = argparse.ArgumentParser(prog="kairos run", description="Run a KAIROS pipeline.")
    sub = parser.add_subparsers(dest="pipeline", required=True)
    weekly = sub.add_parser("weekly", help="the full weekly pipeline")
    weekly.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    weekly.add_argument("--cpu-only", action="store_true", help="never use the GPU (CI runners)")
    weekly.add_argument("--dry-run", action="store_true", help="resolve stages without running")
    weekly.add_argument("--only", action="append", default=[], metavar="STAGE")
    weekly.add_argument("--skip", action="append", default=[], metavar="STAGE")
    weekly.add_argument("--keep-going", action="store_true", help="continue after a failure")
    weekly.add_argument(
        "--strict", action="store_true", help="exit 2 if any selected stage is unavailable"
    )
    weekly.add_argument("--report", type=Path, help="write the JSON run report here")
    weekly.add_argument("--as-of", help="ISO date for the run (default: now, UTC)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Exit codes: 0 ok, 1 a stage failed, 2 strict and a stage unavailable."""
    args = build_parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    ctx = RunContext(
        run_id=uuid.uuid4().hex[:12],
        as_of=as_of,
        root=args.root.resolve(),
        cpu_only=args.cpu_only,
        dry_run=args.dry_run,
    )
    try:
        report = run_pipeline(
            ctx, WEEKLY_STAGES, only=args.only, skip=args.skip, keep_going=args.keep_going
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report.to_dict(), indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    for r in report.results:
        print(f"{r.name:<9} {r.status:<12} {r.seconds:7.2f}s  {r.detail}")
    if report.failed:
        return 1
    if args.strict and report.unavailable:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
