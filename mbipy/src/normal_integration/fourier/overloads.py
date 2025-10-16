"""Numba overloads for normal integration functions using Fourier methods."""

from __future__ import annotations

__all__ = (
    "_dct2_2d_overload",
    "_dst1_2d_overload",
    "_fft_2d_overload",
    "_flip_overload",
    "_idct2_2d_overload",
    "_idst1_2d_overload",
    "_ifft_2d_overload",
    "_irfft_2d_overload",
    "_rfft_2d_overload",
)

import importlib

import numpy as np
from numba import extending, types
from numba.core import errors

from mbipy.src.config import config as cfg
from mbipy.src.normal_integration.fourier import (
    arnison,
    dct_poisson,
    dst_poisson,
    frankot,
    kottler,
)
from mbipy.src.normal_integration.fourier.padding import antisymmetric, flip
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

extending.register_jitable(antisymmetric)
extending.register_jitable(arnison)
extending.register_jitable(dct_poisson)
extending.register_jitable(dst_poisson)
extending.register_jitable(frankot)
extending.register_jitable(kottler)


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
def _rfft_2d_overload(
    x: types.Array,
    s: types.UniTuple,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
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
def _irfft_2d_overload(
    x: types.Array,
    s: types.UniTuple,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
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
def _fft_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    _check_x_workers(x, workers)
    axes = (-2, -1)

    def impl(
        x: types.Array,
        workers: types.Integer | types.NoneType = None,
    ) -> types.Array:
        return np.fft.fftn(x, axes=axes, workers=workers)

    return impl


@extending.overload(ifft_2d)
def _ifft_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    _check_x_workers(x, workers)
    axes = (-2, -1)

    def impl(
        x: types.Array,
        workers: types.Integer | types.NoneType = None,
    ) -> types.Array:
        return np.fft.ifftn(x, axes=axes, workers=workers)

    return impl


@extending.overload(dct2_2d)
def _dct2_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    _check_x_workers(x, workers)
    axes = (-2, -1)
    if cfg.have_scipy:
        fft = importlib.import_module("scipy.fft")

        def impl(
            x: types.Array,
            workers: types.Integer | types.NoneType = None,
        ) -> types.Array:
            return fft.dctn(x, type=2, axes=axes, workers=workers)

        return impl
    msg = "Scipy is required for the DCT"
    raise errors.NumbaNotImplementedError(msg)


@extending.overload(idct2_2d)
def _idct2_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    _check_x_workers(x, workers)
    axes = (-2, -1)
    if cfg.have_scipy:
        fft = importlib.import_module("scipy.fft")

        def impl(
            x: types.Array,
            workers: types.Integer | types.NoneType = None,
        ) -> types.Array:
            return fft.idctn(x, type=2, axes=axes, workers=workers)

        return impl
    msg = "Scipy is required for the IDCT"
    raise errors.NumbaNotImplementedError(msg)


@extending.overload(dst1_2d)
def _dst1_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    _check_x_workers(x, workers)
    axes = (-2, -1)
    if cfg.have_scipy:
        fft = importlib.import_module("scipy.fft")

        def impl(
            x: types.Array,
            workers: types.Integer | types.NoneType = None,
        ) -> types.Array:
            return fft.dstn(x, type=1, axes=axes, workers=workers)

        return impl
    msg = "Scipy is required for the DST"
    raise errors.NumbaNotImplementedError(msg)


@extending.overload(idst1_2d)
def _idst1_2d_overload(
    x: types.Array,
    workers: types.Integer | types.NoneType = None,
) -> types.Array:
    _check_x_workers(x, workers)
    axes = (-2, -1)
    if cfg.have_scipy:
        fft = importlib.import_module("scipy.fft")

        def impl(
            x: types.Array,
            workers: types.Integer | types.NoneType = None,
        ) -> types.Array:
            return fft.idstn(x, type=1, axes=axes, workers=workers)

        return impl
    msg = "Scipy is required for the IDST"
    raise errors.NumbaNotImplementedError(msg)


@extending.overload(flip)
def _flip_overload(
    m: types.Array,
    axis: types.Integer | types.UniTuple | types.NoneType = None,
) -> types.Array:
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
