"""Utility functions for normal integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy import broadcast_shapes

from mbipy.src.config import __have_numba__

if TYPE_CHECKING:

    from numpy.typing import DTypeLike, NDArray


def check_shapes(*arrays: NDArray) -> tuple[int, int]:
    """Check the shapes of arrays broadcast, returning the last two dimensions.

    Returns
    -------
    tuple[int, int]
        Shape of the last two dimensions of the broadcasted arrays.

    """
    return broadcast_shapes(*(i.shape for i in arrays))[-2:]


if __have_numba__:
    import numba as nb
    from numba import types
    from numba.core import errors

    _numba2numpy = {
        types.float32: np.float32,
        types.float64: np.float64,
        types.complex64: np.complex64,
        types.complex128: np.complex128,
        nb.typeof(np.float32): np.float32,
        nb.typeof(np.float64): np.float64,
        nb.typeof(np.complex64): np.complex64,
        nb.typeof(np.complex128): np.complex128,
    }

    _numpy2numba = {
        np.dtype(np.float32): types.float32,
        np.dtype(np.float64): types.float64,
        np.dtype(np.complex64): types.complex64,
        np.dtype(np.complex128): types.complex128,
    }

    @nb.extending.overload(check_shapes)
    def overload_check_shapes(gy: tuple[int, ...], gx: tuple[int, ...]) -> Callable:
        """Numba compatible implementation of check_shapes.

        Parameters
        ----------
        gy : tuple[int, ...]
            Shape of the vertical gradient.
        gx : tuple[int, ...]
            Shape of the horizontal gradient.

        Returns
        -------
        Callable
            Numba implementation of check_shapes.

        """
        if not isinstance(gy, types.Array) or not isinstance(gx, types.Array):
            msg = f"Both arguments must be arrays. Got {gy} and {gx}."
            raise errors.NumbaTypeError(msg)

        def impl(gy: tuple[int, ...], gx: tuple[int, ...]) -> tuple[int, int]:
            return np.broadcast_shapes(gy.shape, gx.shape)[-2:]

        return impl

    # # TODO(nin17): result_type func to avoid overloading numpy function
    @nb.extending.overload(np.result_type)
    def overload_result_type(*arrays: NDArray | DTypeLike) -> Callable:

        dtypes = [
            _numba2numpy[i.dtype if isinstance(i, types.Array) else i] for i in arrays
        ]
        dtype = _numpy2numba[np.result_type(*dtypes)]

        def impl(*arrays: NDArray | DTypeLike) -> DTypeLike:
            return dtype

        return impl

    # if not hasattr(np, "astype"):  # ???
    #     np.astype = lambda x, dtype, copy: x.astype(dtype)

    # # TODO(nin17): astype func to avoid overloading numpy function
    @nb.extending.overload(np.astype)
    def overload_as_type(x, dtype, copy) -> Callable:  # noqa: ANN001
        def impl(x, dtype, copy):  # noqa: ANN001
            return x.astype(dtype)

        return impl

    # if not hasattr(np, "isdtype"):

    #     def isdtype(dtype, kind):
    #         if kind != "real floating":
    #             msg = "Invalid kind."
    #             raise ValueError(msg)
    #         return dtype.kind == "f"

    #     np.isdtype = isdtype

    # # TODO(nin17): isdtype func to avoid overloading numpy function
    @nb.extending.overload(np.isdtype)
    def overload_isdtype(dtype, kind):

        if not isinstance(dtype.dtype, types.Float):
            raise ValueError("Invalid dtype.")

        def impl(dtype, kind):

            return True

        return impl
