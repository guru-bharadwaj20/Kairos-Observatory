"""Execute the tutorial notebooks' code cells in order (CPU, headless)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("numpy")

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_three_tutorials_exist() -> None:
    assert [p.name[:2] for p in NOTEBOOKS] == ["01", "02", "03"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    namespace: dict[str, object] = {"__name__": "__main__"}
    for cell in book.cells:
        if cell.cell_type == "code":
            assert not cell.get("outputs"), "commit notebooks with outputs cleared"
            exec(compile(cell.source, f"{path.name}", "exec"), namespace)
    assert capsys.readouterr().out
