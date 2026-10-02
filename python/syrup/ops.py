"""Operations by name.

`from syrup.ops import find_faces_in_top_half` resolves the name at import
time (module __getattr__, PEP 562) and compiles the operation on its first
call. A name Syrup cannot honour raises syrup.IntentError at the import.
"""

from . import resolve as _resolve

__all__ = []


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _resolve(name)
