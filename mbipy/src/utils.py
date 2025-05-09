"""_summary_"""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_compat as compat
from array_api_compat import (
    is_cupy_namespace,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)

if TYPE_CHECKING:  # pragma: no cover
    from types import ModuleType

    from numpy.typing import NDArray

__all__ = (
    "array_namespace",
    "is_cupy_namespace",
    "is_jax_namespace",
    "is_numpy_namespace",
    "is_torch_namespace",
)


def array_namespace(*arrays: NDArray) -> ModuleType:
    """Get the array namespace of the arrays.

    Parameters
    ----------
    arrays : NDArray
        The arrays to check.

    Returns
    -------
    ModuleType
        The array namespace of the arrays.
    """
    return compat.array_namespace(*arrays, api_version="2024.12")


try:
    from simple_pytree import Pytree, static_field

except ImportError:

    class Pytree:
        def __init_subclass__(cls, mutable: bool = False):
            super().__init_subclass__()

    def static_field(): ...


def setitem(a, i, v):
    xp = array_namespace(a)
    if is_jax_namespace(xp):
        a = a.at[i].set(v)
    else:
        a[i] = v
    return a


def isub(a, i, v):
    xp = array_namespace(a)
    if is_jax_namespace(xp):
        ai = a.at[i]
        a = ai.subtract(v) if hasattr(ai, "subtract") else ai.add(-v)
    else:
        a[i] -= v
    return a


def iadd(a, i, v):
    xp = array_namespace(a)
    if is_jax_namespace(xp):
        a = a.at[i].add(v)
    else:
        a[i] += v
    return a


def idiv(a, i, v):
    xp = array_namespace(a)
    if is_jax_namespace(xp):
        a = a.at[i].divide(v)
    else:
        a[i] /= v
    return a


def imul(array, indices, values):
    xp = array_namespace(array)
    if is_jax_namespace(xp):
        array = array.at[indices].multiply(values)
    else:
        array[indices] *= values
    return array


def cast_scalar(scalar, dtype):
    # !!! Dummy function - only needed for it's numba overload
    return scalar


def get_dtypes(*arrays):
    xp = array_namespace(*arrays)
    dtype = xp.result_type(*arrays)
    if not xp.isdtype(dtype, "real floating"):
        msg = "Input arrays must be real-valued."
        raise ValueError(msg)
    return dtype, xp.result_type(dtype, xp.complex64)


def astype(array, dtype, copy=False):
    # !!! To avoid numba overload of numpy function
    xp = array_namespace(array)
    return xp.astype(array, dtype, copy=copy)
