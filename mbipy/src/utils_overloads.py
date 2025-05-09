"""Numba overloads for functions in utils.py."""

from __future__ import annotations

__all__ = (
    "_overload_array_namespace",
    "_overload_astype",
    "_overload_cast_scalar",
    "_overload_get_dtypes",
    "_overload_idiv",
    "_overload_imul",
    "_overload_isub",
    "_overload_setitem",
)

import numpy as np
from numba import extending, typeof, types
from numba.core import errors

from mbipy.src.utils import (
    array_namespace,
    astype,
    cast_scalar,
    get_dtypes,
    idiv,
    imul,
    isub,
    setitem,
)

_numba2numpy = {
    types.int8: np.int8,
    types.int16: np.int16,
    types.int32: np.int32,
    types.int64: np.int64,
    types.uint8: np.uint8,
    types.uint16: np.uint16,
    types.uint32: np.uint32,
    types.uint64: np.uint64,
    types.bool_: np.bool_,
    types.float32: np.float32,
    types.float64: np.float64,
    types.complex64: np.complex64,
    types.complex128: np.complex128,
    typeof(np.float32): np.float32,
    typeof(np.float64): np.float64,
    typeof(np.complex64): np.complex64,
    typeof(np.complex128): np.complex128,
}

_numpy2numba = {
    np.dtype(np.float32): np.float32,
    np.dtype(np.float64): np.float64,
    np.dtype(np.complex64): np.complex64,
    np.dtype(np.complex128): np.complex128,
}


@extending.overload(setitem)
def _overload_setitem(array: types.Array, indices, v) -> types.Array:
    if not isinstance(array, types.Array):
        msg = "Array must be of type Array."
        raise errors.NumbaTypeError(msg)

    def impl(array: types.Array, indices, v) -> types.Array:  # pragma: no cover
        array[indices] = v
        return array

    return impl


@extending.overload(isub)
def _overload_isub(array: types.Array, indices, v) -> types.Array:
    if not isinstance(array, types.Array):
        msg = "Array must be of type Array."
        raise errors.NumbaTypeError(msg)

    if isinstance(indices, types.EllipsisType):

        def impl(array: types.Array, indices, v) -> types.Array:
            array -= v
            return array

    else:

        def impl(array: types.Array, indices, v) -> types.Array:  # pragma: no cover
            array[indices] -= v
            return array

    return impl


@extending.overload(idiv)
def _overload_idiv(array: types.Array, indices, v) -> types.Array:
    if not isinstance(array, types.Array):
        msg = "Array must be of type Array."
        raise errors.NumbaTypeError(msg)

    if isinstance(indices, types.EllipsisType):

        def impl(array: types.Array, indices, v) -> types.Array:  # pragma: no cover
            array /= v
            return array

    else:

        def impl(array: types.Array, indices, v) -> types.Array:  # pragma: no cover
            array[indices] /= v
            return array

    return impl


@extending.overload(imul)
def _overload_imul(array: types.Array, indices, v) -> types.Array:
    if not isinstance(array, types.Array):
        msg = "Array must be of type Array."
        raise errors.NumbaTypeError(msg)

    if isinstance(indices, types.EllipsisType):

        def impl(array: types.Array, indices, v) -> types.Array:
            array *= v
            return array

    else:

        def impl(array: types.Array, indices, v) -> types.Array:  # pragma: no cover
            array[indices] *= v
            return array

    return impl


@extending.overload(array_namespace)
def _overload_array_namespace(*arrays: types.Array) -> types.Module:
    if not all(isinstance(i, types.Array) for i in arrays):
        msg = "All arguments must be of type Array."
        raise errors.NumbaTypeError(msg)

    import numpy as np

    def impl(*arrays: types.Array) -> types.Module:  # pragma: no cover
        return np

    return impl


@extending.overload(cast_scalar)
def _overload_cast_scalar(
    scalar: types.Number,
    dtype: types.NumberClass,
) -> types.Number:
    if not isinstance(scalar, types.Number):
        msg = "Scalar must be of type Number."
        raise errors.NumbaTypeError(msg)
    if not isinstance(dtype, types.NumberClass):
        msg = "Dtype must be of type Dtype."
        raise errors.NumbaTypeError(msg)

    def impl(scalar: types.Number, dtype: types.NumberClass) -> types.Number:
        return dtype(scalar)  # pragma: no cover

    return impl


@extending.overload(get_dtypes)
def _overload_get_dtypes(*arrays: types.Array) -> tuple[types.DType, types.DType]:
    if not all(isinstance(i, types.Array) for i in arrays):
        msg = "All arguments must be of type Array."
        raise errors.NumbaTypeError(msg)

    dtypes = [_numba2numpy[i.dtype] for i in arrays]
    dtype = np.result_type(*dtypes)
    cdtype = np.result_type(dtype, np.complex64)

    if dtype.kind != "f":
        msg = f"Input arrays must be real-valued. Got {dtype}."
        raise errors.NumbaTypeError(msg)

    dtype = _numpy2numba[dtype]
    cdtype = _numpy2numba[cdtype]

    def impl(*arrays: types.Array) -> tuple[types.DType, types.DType]:

        return dtype, cdtype  # pragma: no cover

    return impl


@extending.overload(astype)
def _overload_astype(
    x: types.Array,
    dtype: types.DType,
    copy: bool = True,
) -> types.Array:
    if not isinstance(x, types.Array):
        msg = "x must be of type Array."
        raise errors.NumbaTypeError(msg)
    # if not isinstance(dtype, types.DType):
    #     msg = "dtype must be of type Dtype."
    #     raise errors.NumbaTypeError(msg)

    def impl(x: types.Array, dtype: types.DType, copy: bool = True) -> types.Array:
        return x.astype(dtype)  # pragma: no cover

    return impl
