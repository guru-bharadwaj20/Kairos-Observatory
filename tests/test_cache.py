import numpy as np
import pandas as pd
import pytest

from kairos.compute import cache as cache_mod
from kairos.compute.cache import cached_fit, clear_fit_cache, data_hash, get_fit_cache

CALLS: list[tuple] = []


@cached_fit(version="1", ignore=["verbose"])
def fake_fit(df, scale, verbose=False):
    CALLS.append((len(df), scale))
    return {"mean": float(df["y"].mean() * scale)}


@cached_fit
def bare(x):
    CALLS.append(("bare", x))
    return x + 1


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CACHE_DIR", str(tmp_path / "cache"))
    CALLS.clear()
    return tmp_path / "cache"


def frame(values):
    return pd.DataFrame(
        {"ds": pd.date_range("2020-01-01", periods=len(values), freq="MS"), "y": values}
    )


def test_rerun_skips_finished_fit():
    df = frame([1.0, 2.0, 3.0])
    first = fake_fit(df, 2.0)
    assert fake_fit.is_cached(df, 2.0)
    second = fake_fit(df.copy(), 2.0, verbose=True)  # equal data, ignored arg
    assert first == second == {"mean": 4.0}
    assert len(CALLS) == 1


def test_positional_and_keyword_calls_share_a_key():
    df = frame([1.0])
    assert fake_fit.key(df, 2.0) == fake_fit.key(df=df, scale=2.0)
    assert fake_fit.key(df, 2.0) == fake_fit.key(df, 2.0, verbose=True)


def test_changed_data_or_params_refit():
    fake_fit(frame([1.0, 2.0]), 1.0)
    fake_fit(frame([1.0, 2.5]), 1.0)
    fake_fit(frame([1.0, 2.0]), 3.0)
    assert len(CALLS) == 3


def test_code_version_namespaces_cache(tmp_cache):
    assert get_fit_cache("1").location != get_fit_cache("2").location
    assert "kairos-" in str(get_fit_cache("1").location)
    assert str(get_fit_cache("1").location).startswith(str(tmp_cache))


def test_bare_decorator_and_clear(tmp_cache):
    assert bare(1) == 2
    assert bare(1) == 2
    assert len(CALLS) == 1
    bare.clear()
    assert not bare.is_cached(1)
    bare(1)
    assert len(CALLS) == 2
    clear_fit_cache()
    assert not (tmp_cache / "fits").exists()


def test_cache_survives_new_process_state():
    df = frame([5.0])
    fake_fit(df, 1.0)
    fake_fit._memorized.clear()  # simulate a fresh interpreter
    cache_mod._memory.cache_clear()
    fake_fit(df, 1.0)
    assert len(CALLS) == 1


def test_lazy_module_attribute():
    assert cache_mod.fit_cache.location == get_fit_cache().location
    with pytest.raises(AttributeError):
        _ = cache_mod.nope


def test_data_hash_content_based():
    a = frame([1.0, 2.0])
    b = frame(np.array([1.0, 2.0]))
    assert data_hash(a) == data_hash(b)
    assert data_hash(a) != data_hash(frame([1.0, 2.1]))
    assert data_hash(a) != data_hash(a.rename(columns={"y": "z"}))
    assert data_hash({"k": 1}) == data_hash({"k": 1})
