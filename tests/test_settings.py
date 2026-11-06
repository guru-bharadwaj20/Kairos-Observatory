from pathlib import Path

import pytest
from pydantic import ValidationError

from kairos.settings import DEFAULT_SEED, Settings, get_settings


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for var in ("KAIROS_N_JOBS", "KAIROS_DEVICE", "KAIROS_CONFIG", "KAIROS_SEED"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.n_jobs == 28
    assert s.device == "auto"
    assert s.seed == DEFAULT_SEED == 20261113
    assert s.data_dir == Path("data")
    assert s.cache_dir == Path(".cache/kairos")
    assert s.vram_budget_bytes == 20 * 1024**3


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("KAIROS_N_JOBS", "4")
    monkeypatch.setenv("KAIROS_DEVICE", "CPU")
    monkeypatch.setenv("KAIROS_LOG_LEVEL", "debug")
    s = get_settings()
    assert (s.n_jobs, s.device, s.log_level) == (4, "cpu", "DEBUG")


def test_yaml_file_and_precedence(monkeypatch, tmp_path):
    cfg = tmp_path / "k.yaml"
    cfg.write_text("n_jobs: 6\nseed: 7\ndata_dir: /srv/kairos\n")
    monkeypatch.setenv("KAIROS_CONFIG", str(cfg))
    monkeypatch.setenv("KAIROS_SEED", "11")
    monkeypatch.delenv("KAIROS_N_JOBS", raising=False)
    s = Settings()
    assert s.n_jobs == 6  # from YAML
    assert s.seed == 11  # env beats YAML
    assert s.data_dir == Path("/srv/kairos")
    assert Settings(seed=3).seed == 3  # init beats env


def test_local_kairos_yaml_is_picked_up(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KAIROS_CONFIG", raising=False)
    monkeypatch.delenv("KAIROS_N_JOBS", raising=False)
    (tmp_path / "kairos.yaml").write_text("n_jobs: 3\n")
    assert Settings().n_jobs == 3


@pytest.mark.parametrize(
    "kwargs", [{"n_jobs": 0}, {"device": "tpu"}, {"seed": -1}, {"log_level": "LOUD"}]
)
def test_invalid_values(kwargs):
    with pytest.raises(ValidationError):
        Settings(**kwargs)


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
