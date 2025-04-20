"""Numba overloads for normal integration functions using Fourier methods."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numba import extending, types
from numba.core import errors

from mbipy.src.config import __have_scipy__
from mbipy.src.normal_integration.fourier.padding import antisym, flip
from mbipy.src.normal_integration.fourier.utils import (
    dct2_2d,
    dst1_2d,
    fft_2d,
    idct2_2d,
    idst1_2d,
    ifft_2d,
    irfft_2d,
    rfft_2d,
)

extending.register_jitable(antisym)


def _check_x_workers(x: types.Array, workers: types.Integer | types.NoneType) -> None:
    if not isinstance(x, types.Array):
        msg = f"x must be an array, got {x}."
        raise errors.NumbaTypeError(msg)
    if not isinstance(workers, (types.Integer, types.NoneType)):
        msg = f"workers must be an int or None, got {workers}."
        raise errors.NumbaTypeError(msg)


def _check_s(s: types.UniTuple) -> None:
    if not isinstance(s, types.UniTuple):
        msg = f"s must be a tuple, got {s}."
        raise errors.NumbaTypeError(msg)
    if s.count != 2:  # noqa: PLR2004
        msg = f"s must be a tuple of length 2, got {s}."
        raise errors.NumbaTypeError(msg)
    if not isinstance(s.dtype, types.Integer):
        msg = f"s must be a tuple of integers, got {s}."
        raise errors.NumbaTypeError(msg)


@extending.overload(rfft_2d)
def rfft_2d_overload(
    x: types.Array,
    s: types.UniTuple,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    """Overload rfft_2d for numba."""
    _check_x_workers(x, workers)
    _check_s(s)
    axes = (-2, -1)

    # !!! workers in both numpy and scipy with rocket-fft
    def impl(
        x: types.Array,
        s: types.UniTuple,
        workers: types.Integer | types.NoneType = None,
    ) -> types.Array:
        return np.fft.rfftn(x, s=s, axes=axes, workers=workers)

    return impl


@extending.overload(irfft_2d)
def irfft_2d_overload(
    x: types.Array,
    s: types.UniTuple,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    """Overload irfft_2d for numba."""
    _check_x_workers(x, workers)
    _check_s(s)
    axes = (-2, -1)

    # !!! workers in both numpy and scipy with rocket-fft
    def impl(
        x: types.Array,
        s: types.UniTuple,
        workers: types.Integer | types.NoneType = None,
    ) -> types.Array:
        return np.fft.irfftn(x, s=s, axes=axes, workers=workers)

    return impl


@extending.overload(fft_2d)
def fft_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    """Overload fft_2d for numba."""
    _check_x_workers(x, workers)
    axes = (-2, -1)

    def impl(
        x: types.Array,
        workers: types.Integer | types.NoneType = None,
    ) -> types.Array:
        return np.fft.fftn(x, axes=axes, workers=workers)

    return impl


@extending.overload(ifft_2d)
def ifft_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    """Overload ifft_2d for numba."""
    _check_x_workers(x, workers)
    axes = (-2, -1)

    def impl(
        x: types.Array,
        workers: types.Integer | types.NoneType = None,
    ) -> types.Array:
        return np.fft.ifftn(x, axes=axes, workers=workers)

    return impl


@extending.overload(dct2_2d)
def dct2_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    """Overload dct2_2d for numba."""
    _check_x_workers(x, workers)
    axes = (-2, -1)
    if __have_scipy__:
        from scipy import fft as spfft

        def impl(
            x: types.Array,
            workers: types.Integer | types.NoneType = None,
        ) -> types.Array:
            return spfft.dctn(x, type=2, axes=axes, workers=workers)

        return impl
    msg = "Scipy is required for the DCT"
    raise errors.NumbaNotImplementedError(msg)


@extending.overload(idct2_2d)
def idct2_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    """Overload idct2_2d for numba."""
    _check_x_workers(x, workers)
    axes = (-2, -1)
    if __have_scipy__:
        from scipy import fft as spfft

        def impl(
            x: types.Array,
            workers: types.Integer | types.NoneType = None,
        ) -> types.Array:
            return spfft.idctn(x, type=2, axes=axes, workers=workers)

        return impl
    msg = "Scipy is required for the IDCT"
    raise errors.NumbaNotImplementedError(msg)


@extending.overload(dst1_2d)
def dst1_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    """Overload dst1_2d for numba."""
    _check_x_workers(x, workers)
    axes = (-2, -1)
    if __have_scipy__:
        from scipy import fft as spfft

        def impl(
            x: types.Array,
            workers: types.Integer | types.NoneType = None,
        ) -> types.Array:
            return spfft.dstn(x, type=1, axes=axes, workers=workers)

        return impl
    msg = "Scipy is required for the DST"
    raise errors.NumbaNotImplementedError(msg)


@extending.overload(idst1_2d)
def idst1_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    """Overload idst1_2d for numba."""
    _check_x_workers(x, workers)
    axes = (-2, -1)
    if __have_scipy__:
        from scipy import fft as spfft

        def impl(
            x: types.Array,
            workers: types.Integer | types.NoneType = None,
        ) -> types.Array:
            return spfft.idstn(x, type=1, axes=axes, workers=workers)

        return impl
    msg = "Scipy is required for the IDST"
    raise errors.NumbaNotImplementedError(msg)


@extending.overload(flip)
def flip_overload(
    m: types.Array,
    axis: types.Integer | types.UniTuple | types.NoneType = None,
) -> Callable:
    """Overload flip for numba."""
    if not isinstance(m, types.Array):
        msg = f"a must be an array, got {m}."
        raise errors.NumbaTypeError(msg)
    if isinstance(axis, types.Integer):

        def impl(
            m: types.Array,
            axis: types.Integer | types.UniTuple | types.NoneType = None,
        ) -> types.Array:
            if axis == -1:
                return m[..., ::-1]
            if axis == -2:  # noqa: PLR2004
                return m[..., ::-1, :]
            msg = "Invalid axis."
            raise ValueError(msg)

    elif isinstance(axis, types.UniTuple):

        def impl(
            m: types.Array,
            axis: types.Integer | types.UniTuple | types.NoneType = None,
        ) -> types.Array:
            if axis == (-2, -1):
                return m[..., ::-1, ::-1]
            msg = "Invalid axis"
            raise ValueError(msg)

    elif isinstance(axis, types.NoneType):
        msg = "axis=None is not implemented yet"
        raise errors.NumbaNotImplementedError(msg)

    else:
        valid_types = "types.Integer or types.UniTuple"
        msg = f"Invalid axis type: {axis}. Should be {valid_types}."
        raise errors.NumbaTypeError(msg)
    return impl
