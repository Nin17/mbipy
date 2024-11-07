import importlib
from types import ModuleType

import array_api_compat as compat

from .config import __have_numba__


def array_namespace(*arrays) -> ModuleType:  # noqa: ANN002
    try:
        xp = compat.array_namespace(*arrays)
        if "jax" in xp.__name__:
            return importlib.import_module("jax.numpy")
        return xp
    except TypeError:
        try:
            tnp = importlib.import_module("tensorflow.experimental.numpy")
            tf = importlib.import_module("tensorflow")

            if all(isinstance(i, tf.Tensor) for i in arrays):
                xp = tnp
                xp.experimental_enable_numpy_behavior()
                return xp
        except ImportError:
            ...

        raise


try:
    from simple_pytree import Pytree, static_field

except ImportError:

    class Pytree:
        def __init_subclass__(cls, mutable: bool = False):
            super().__init_subclass__()

    def static_field(): ...


def _isjax(xp):
    return "jax" in xp.__name__


def setitem(a, i, v, xp):
    if _isjax(xp):
        a = a.at[i].set(v)
    else:
        a[i] = v
    return a


def isub(a, i, v, xp):
    if _isjax(xp):
        ai = a.at[i]
        a = ai.subtract(v) if hasattr(ai, "subtract") else ai.add(-v)
    else:
        a[i] -= v
    return a


def iadd(a, i, v, xp):
    if _isjax(xp):
        a = a.at[i].add(v)
    else:
        a[i] += v
    return a


if __have_numba__:
    import numba as nb
    import numpy as np
    from numba import types

    @nb.extending.overload(setitem)
    def overload_setitem(a, i, v, xp):
        def impl(a, i, v, xp):
            a[i] = v
            return a

        return impl

    @nb.extending.overload(isub)
    def overload_isub(a, i, v, xp):
        def impl(a, i, v, xp):
            a[i] -= v
            return a

        return impl

    @nb.extending.overload(array_namespace)
    def overload_array_namespace(*arrays):
        if not all(isinstance(i, types.Array) for i in arrays):
            raise ValueError("All arguments must be of type Array.")

        def impl(*arrays):
            return np

        return impl
