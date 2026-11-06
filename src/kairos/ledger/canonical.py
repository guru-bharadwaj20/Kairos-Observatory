r"""Canonical JSON serialisation and SHA-256 hashing.

The ledger hashes *bytes*, so every value must have exactly one byte encoding.
:func:`canonical_json` produces it:

* object keys are sorted (by Unicode code point) and must be strings;
* no insignificant whitespace (``,`` and ``:`` separators);
* output is ASCII; non-ASCII characters are ``\\uXXXX``-escaped;
* floats are rounded to :data:`FLOAT_SIG_DIGITS` significant digits and then
  written in Python's shortest round-trip form, always with a decimal point or
  exponent (``1.0``, ``1e-07``), so ``1`` (int) and ``1.0`` (float) stay distinct;
* ``-0.0`` is written as ``0.0``; ``NaN`` and ``±inf`` are rejected;
* ``date``/``datetime`` become ISO 8601 strings (datetimes must be timezone-aware
  and are converted to UTC with a ``Z`` suffix);
* tuples become arrays; pydantic models are dumped (``exclude_none=True``) first,
  so optional fields added later do not change the hash of older records.

Rounding floats to 15 significant digits means that values which differ only in
the last ulp (e.g. from a different BLAS summation order) hash identically.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, Final

from pydantic import BaseModel

__all__ = [
    "FLOAT_SIG_DIGITS",
    "HASH_HEX_LEN",
    "canonical_float",
    "canonical_json",
    "is_sha256_hex",
    "sha256_hex",
]

FLOAT_SIG_DIGITS: Final = 15
"""Significant digits kept for floats before hashing."""

HASH_HEX_LEN: Final = 64


def canonical_float(x: float) -> str:
    """Return the canonical text of a finite float.

    Raises:
        ValueError: If ``x`` is NaN or infinite.
    """
    if not math.isfinite(x):
        raise ValueError(f"non-finite float {x!r} cannot be hashed")
    rounded = float(f"{x:.{FLOAT_SIG_DIGITS}g}")
    if not math.isfinite(rounded):  # rounding up past the largest double
        rounded = x
    if rounded == 0.0:
        return "0.0"
    return repr(rounded)


def _datetime_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"naive datetime {value!r}; ledger timestamps must be timezone-aware")
    utc = value.astimezone(UTC).replace(tzinfo=None)
    return utc.isoformat(timespec="microseconds") + "Z"


def _encode(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        out.append(str(int(value)))
    elif isinstance(value, float):
        out.append(canonical_float(value))
    elif isinstance(value, str):
        out.append(json.dumps(value, ensure_ascii=True))
    elif isinstance(value, datetime):
        out.append(json.dumps(_datetime_text(value)))
    elif isinstance(value, date):
        out.append(json.dumps(value.isoformat()))
    elif isinstance(value, BaseModel):
        _encode(value.model_dump(mode="python", exclude_none=True), out)
    elif isinstance(value, Mapping):
        keys = list(value.keys())
        if not all(isinstance(k, str) for k in keys):
            raise TypeError(f"object keys must be strings, got {keys!r}")
        out.append("{")
        for i, key in enumerate(sorted(keys)):
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=True))
            out.append(":")
            _encode(value[key], out)
        out.append("}")
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _encode(item, out)
        out.append("]")
    else:
        raise TypeError(f"cannot canonicalise value of type {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Serialise ``value`` to canonical JSON bytes (see the module docstring)."""
    out: list[str] = []
    _encode(value, out)
    return "".join(out).encode("ascii")


def sha256_hex(value: Any) -> str:
    """Return the lowercase hex SHA-256 of ``canonical_json(value)``.

    ``bytes`` are hashed as-is.
    """
    data = bytes(value) if isinstance(value, bytes | bytearray) else canonical_json(value)
    return hashlib.sha256(data).hexdigest()


def is_sha256_hex(text: object) -> bool:
    """Return whether ``text`` is a 64-character lowercase hex digest."""
    return (
        isinstance(text, str)
        and len(text) == HASH_HEX_LEN
        and all(c in "0123456789abcdef" for c in text)
    )
