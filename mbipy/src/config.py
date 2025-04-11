"""_summary_"""

__all__ = (
    "__have_jax__",
    "__have_numba__",
    "__have_pywavelets__",
    "__have_scipy__",
    "__have_torch__",
)

import importlib

__have_jax__ = bool(importlib.util.find_spec("jax"))
__have_numba__ = bool(importlib.util.find_spec("numba"))
__have_pywavelets__ = bool(importlib.util.find_spec("pywavelets"))
__have_scipy__ = bool(importlib.util.find_spec("scipy"))
__have_torch__ = bool(importlib.util.find_spec("torch"))
