import importlib
from types import ModuleType

import array_api_compat as compat
from array_api_compat import is_jax_namespace

from .config import __have_numba__


def array_namespace(*arrays) -> ModuleType:  # noqa: ANN002
    try:
        xp = compat.array_namespace(*arrays)
        if is_jax_namespace(xp):
            return importlib.import_module("jax.numpy")
        return xp
    except TypeError:
        # ??? why
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


def cast_scalar(x, dtype):
    # !!! Dummy function - only needed for it's numba overload
    return x


if __have_numba__:
    import numba as nb
    import numpy as np
    from numba import types
    from numba.core import errors

    @nb.extending.overload(setitem)
    def overload_setitem(a, i, v):
        def impl(a, i, v):
            a[i] = v
            return a

        return impl

    @nb.extending.overload(isub)
    def overload_isub(a, i, v):
        def impl(a, i, v):
            a[i] -= v
            return a

        return impl

    @nb.extending.overload(idiv)
    def overload_idiv(a, i, v):
        def impl(a, i, v):
            a[i] /= v
            return a

        return impl

    @nb.extending.overload(array_namespace)
    def overload_array_namespace(*arrays):
        if not all(isinstance(i, types.Array) for i in arrays):
            msg = "All arguments must be of type Array."
            raise errors.NumbaTypeError(msg)

        def impl(*arrays):
            return np

        return impl

    @nb.extending.overload(cast_scalar)
    def overload_cast_scalar(scalar, dtype):
        if not isinstance(scalar, types.Number):
            msg = "Scalar must be of type Number."
            raise errors.NumbaTypeError(msg)
        if not isinstance(dtype, types.NumberClass):
            msg = "Dtype must be of type Dtype."
            raise errors.NumbaTypeError(msg)

        def impl(scalar, dtype):
            return dtype(scalar)

        return impl
