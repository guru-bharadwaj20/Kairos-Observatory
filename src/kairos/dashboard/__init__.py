"""KAIROS public dashboard: renders a published snapshot, nothing else."""

from kairos.dashboard.snapshot import Snapshot, SnapshotError, load_demo, load_snapshot

__all__ = ["Snapshot", "SnapshotError", "load_demo", "load_snapshot"]
