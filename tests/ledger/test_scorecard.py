from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from kairos.cli import app
from kairos.ledger.chain import Ledger
from kairos.ledger.scorecard import load_scores, scorecard, scorecard_markdown
from kairos.ledger.scoring import score_matured

T0 = datetime(2026, 11, 6, 12, tzinfo=UTC)
runner = CliRunner()


def actuals(forecast, factor):
    return pd.DataFrame(
        {
            "ds": pd.to_datetime([p.ds for p in forecast.points]),
            "y": [p.at(0.5) * factor for p in forecast.points],
            "series_id": forecast.series_id,
        }
    )


@pytest.fixture
def scored(tmp_path, make_forecast):
    ledger = Ledger(tmp_path / "forecasts.jsonl")
    hit = make_forecast().with_claims([140.0])
    miss = make_forecast(series_id="metr_horizon", base=10.0)
    ledger.append_forecast(hit, manifest_hash="a" * 64, recorded_at=T0)
    ledger.append_forecast(miss, manifest_hash="a" * 64, recorded_at=T0 + timedelta(1))
    frame = pd.concat([actuals(hit, 1.0), actuals(miss, 3.0)])
    score_matured(ledger, frame, today=date(2027, 5, 1))
    return ledger


def test_scorecard_rows(scored):
    table = scorecard(scored).set_index("series_id")
    assert list(table.index) == ["frontier_compute", "metr_horizon", "ALL"]
    assert table.loc["frontier_compute", "coverage"] == 1.0
    assert table.loc["metr_horizon", "coverage"] == 0.0
    assert table.loc["ALL", "coverage"] == 0.5
    assert table.loc["ALL", "n_points"] == 12
    assert table.loc["frontier_compute", "nominal"] == 0.9
    assert table.loc["frontier_compute", "claims_within"] == 1.0
    assert table.loc["metr_horizon", "pit_hist"] == "0 0 0 0 0 0 0 0 0 6"
    assert np.isnan(table.loc["ALL", "mean_crps"])
    assert table.loc["metr_horizon", "mean_crps"] > table.loc["frontier_compute", "mean_crps"]


def test_superseded_scores_dropped(scored):
    first_score = next(e for e in scored.entries() if e.kind == "score")
    scored.append_correction(first_score.entry_hash, "scored against a bad vintage")
    assert len(load_scores(scored.entries())) == 1


def test_empty_scorecard(tmp_path):
    table = scorecard(Ledger(tmp_path / "x.jsonl"))
    assert table.empty
    assert "No forecasts have matured" in scorecard_markdown(table)


def test_markdown_and_cli(scored, tmp_path):
    md = scorecard_markdown(scorecard(scored), generated=date(2027, 5, 1))
    assert md.startswith("# KAIROS forecast scorecard")
    assert "| frontier_compute | prophet-null | 1 | 6 | 0 |" in md
    out = tmp_path / "docs" / "scorecard.md"
    result = runner.invoke(
        app, ["ledger", "scorecard", "--ledger", str(scored.path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "metr_horizon" in out.read_text()
    printed = runner.invoke(app, ["ledger", "scorecard", "--ledger", str(scored.path)])
    assert "pit_ks" in printed.output
