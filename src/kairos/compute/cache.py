"""Fit cache: never refit a model whose data, parameters and code are unchanged.

Built on :class:`joblib.Memory`. A cached call is keyed by

* **data hash and parameters:** joblib hashes every argument, DataFrames included;
* **code:** joblib also hashes the decorated function's source, and the cache
  directory is namespaced by the ``kairos`` version plus a per-function
  ``version`` tag that you bump when behaviour changes outside the function body.

Usage::

    from kairos.compute.cache import cached_fit

    @cached_fit(version="1")
    def fit_null_model(df: pd.DataFrame, changepoint_prior_scale: float) -> dict: ...

The cache lives under ``Settings.cache_dir / "fits"`` and is resolved on the
first call, so ``KAIROS_CACHE_DIR`` set before that call is honoured. It works
across :func:`kairos.compute.pmap` workers because the store is on disk.
"""

from __future__ import annotations

import functools
import inspect
import shutil
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast, overload

import joblib
import pandas as pd

__all__ = [
    "CachedFit",
    "cache_root",
    "cached_fit",
    "clear_fit_cache",
    "data_hash",
    "get_fit_cache",
]

_lock = threading.Lock()


def cache_root(cache_dir: Path | str | None = None) -> Path:
    """Directory holding all cached fits (``<cache_dir>/fits``)."""
    if cache_dir is None:
        from kairos.settings import get_settings

        cache_dir = get_settings().cache_dir
    return Path(cache_dir) / "fits"


def _code_namespace(version: str) -> str:
    from kairos import __version__

    safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in __version__)
    return f"kairos-{safe}__v{version}"


@functools.lru_cache(maxsize=32)
def _memory(location: str) -> joblib.Memory:
    return joblib.Memory(location=location, verbose=0, compress=3)


def get_fit_cache(version: str = "0", cache_dir: Path | str | None = None) -> joblib.Memory:
    """Return the :class:`joblib.Memory` for code version ``version``."""
    return _memory(str(cache_root(cache_dir) / _code_namespace(version)))


def __getattr__(name: str) -> Any:
    # ``fit_cache`` is resolved lazily so importing this module never reads settings.
    if name == "fit_cache":
        return get_fit_cache()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def data_hash(obj: Any) -> str:
    """Stable content hash of a DataFrame, array or any picklable object.

    DataFrames are hashed by values, index, column names and dtypes, so two frames
    with equal content hash equally regardless of memory layout.
    """
    if isinstance(obj, pd.DataFrame):
        values = pd.util.hash_pandas_object(obj, index=True).to_numpy()
        meta = (list(map(str, obj.columns)), [str(t) for t in obj.dtypes])
        return cast(str, joblib.hash((values, meta)))
    return cast(str, joblib.hash(obj))


def _run_fit(
    key: str, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Execute a fit; joblib keys the stored result on ``key`` only."""
    return fn(*args, **kwargs)


def _normalise(value: Any) -> Any:
    if isinstance(value, pd.DataFrame | pd.Series):
        frame = value.to_frame() if isinstance(value, pd.Series) else value
        return ("__frame__", data_hash(frame))
    return value


def _source_hash(fn: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = getattr(getattr(fn, "__code__", None), "co_code", b"").hex()
    return cast(str, joblib.hash(source))


class CachedFit[**P, R]:
    """A function whose results are memoised on disk; see :func:`cached_fit`.

    The cache key is the hash of: the function's qualified name and source, the
    bound arguments (defaults applied, ``ignore`` names dropped, DataFrames hashed
    by content via :func:`data_hash`). Equal data therefore hits the cache even
    when it is a different object or memory layout.
    """

    def __init__(self, fn: Callable[P, R], version: str, ignore: Iterable[str] | None) -> None:
        functools.update_wrapper(self, fn)
        self.fn = fn
        self.version = version
        self.ignore = frozenset(ignore or ())
        self._signature = inspect.signature(fn)
        self._code = f"{fn.__module__}.{fn.__qualname__}:{_source_hash(fn)}"
        self._memorized: dict[str, Any] = {}

    def key(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Return the cache key for a call."""
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        items = sorted(
            (name, _normalise(value))
            for name, value in bound.arguments.items()
            if name not in self.ignore
        )
        return cast(str, joblib.hash((self._code, items)))

    def _bound(self) -> Any:
        memory = get_fit_cache(self.version)
        location = str(memory.location)
        with _lock:
            if location not in self._memorized:
                self._memorized[location] = memory.cache(_run_fit, ignore=["fn", "args", "kwargs"])
            return self._memorized[location]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Return the stored result for this call, computing and storing it if needed."""
        key = self.key(*args, **kwargs)
        return cast(R, self._bound()(key, self.fn, args, kwargs))

    def is_cached(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        """Whether this exact call already has a stored result."""
        key = self.key(*args, **kwargs)
        return bool(self._bound().check_call_in_cache(key, self.fn, args, kwargs))

    def clear(self) -> None:
        """Delete every stored fit of the current code version (all decorated functions)."""
        self._bound().clear(warn=False)


@overload
def cached_fit[**P, R](fn: Callable[P, R], /) -> CachedFit[P, R]: ...
@overload
def cached_fit[**P, R](
    *, version: str = "0", ignore: Iterable[str] | None = None
) -> Callable[[Callable[P, R]], CachedFit[P, R]]: ...
def cached_fit[**P, R](
    fn: Callable[P, R] | None = None,
    /,
    *,
    version: str = "0",
    ignore: Iterable[str] | None = None,
) -> CachedFit[P, R] | Callable[[Callable[P, R]], CachedFit[P, R]]:
    """Decorate a fit function so reruns with identical inputs are loaded from disk.

    Args:
        fn: Function to cache (when used as ``@cached_fit`` without parentheses).
        version: Bump to invalidate results when behaviour changes in code the
            function calls but does not contain.
        ignore: Argument names excluded from the key (e.g. ``"verbose"``, ``"n_jobs"``).

    Returns:
        A :class:`CachedFit` wrapper, or a decorator producing one.

    Note:
        Decorated functions must be defined at module level and return picklable
        results.
    """

    def decorate(f: Callable[P, R]) -> CachedFit[P, R]:
        return CachedFit(f, version, ignore)

    return decorate(fn) if fn is not None else decorate


def clear_fit_cache(cache_dir: Path | str | None = None) -> None:
    """Remove every cached fit for every code version."""
    root = cache_root(cache_dir)
    _memory.cache_clear()
    if root.exists():
        shutil.rmtree(root)
