"""Utilities for normal integration methods that use DFT, DCT or DST transforms."""

from __future__ import annotations

import importlib

from array_api_compat import (
    is_cupy_namespace,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)
from numpy.lib.array_utils import normalize_axis_index

from mbipy.src.config import __have_numba__, __have_scipy__
from mbipy.src.utils import array_namespace, idiv


def fft2(a, s=None, workers=None, use_rfft=True):
    axes = (-2, -1)
    xp = array_namespace(a)
    if is_numpy_namespace(xp) and __have_scipy__:
        fft = importlib.import_module("scipy.fft")
        _fft2 = fft.rfftn if use_rfft else fft.fftn
        return _fft2(a, s=s, axes=axes, workers=workers)

    _fft2 = xp.fft.rfftn if use_rfft else xp.fft.fftn
    return _fft2(a, s=s, axes=axes)


def ifft2(a, s=None, workers=None, use_rfft=True):
    axes = (-2, -1)
    xp = array_namespace(a)
    if is_numpy_namespace(xp) and __have_scipy__:
        fft = importlib.import_module("scipy.fft")
        _ifft2 = fft.irfftn if use_rfft else fft.ifftn
        return _ifft2(a, s=s, axes=axes, workers=workers)
    _ifft2 = xp.fft.irfftn if use_rfft else xp.fft.ifftn
    return _ifft2(a, s=s, axes=axes)


def dct2(x, workers=None):
    xp = array_namespace(x)
    kwargs = {"workers": workers, "axes": (-2, -1)}
    if is_numpy_namespace(xp):
        if not __have_scipy__:
            msg = "Need SciPy for the DCT"
            raise ImportError(msg)
        dctn = importlib.import_module("scipy.fft").dctn
    else:
        del kwargs["workers"]
        if is_cupy_namespace(xp):
            # TODO(nin17): optional pyvkfft
            dctn = importlib.import_module("cupyx.scipy.fft").dctn
        elif is_jax_namespace(xp):
            dctn = importlib.import_module("jax.scipy.fft").dctn
        elif is_torch_namespace(xp):
            dctn = importlib.import_module("torch_dct").dct_2d
            del kwargs["axes"]
        else:
            msg = "dct is not implemented for this array namespace."
            raise NotImplementedError(msg)
    return dctn(x, **kwargs)


def idct2(x, workers=None):
    xp = array_namespace(x)
    kwargs = {"workers": workers, "axes": (-2, -1)}
    if is_numpy_namespace(xp):
        if not __have_scipy__:
            msg = "Need SciPy for the IDCT"
            raise ImportError(msg)
        idctn = importlib.import_module("scipy.fft").idctn
    else:
        del kwargs["workers"]
        if is_cupy_namespace(xp):
            # TODO(nin17): optional pyvkfft
            idctn = importlib.import_module("cupyx.scipy.fft").idctn
        elif is_jax_namespace(xp):
            idctn = importlib.import_module("jax.scipy.fft").idctn
        elif is_torch_namespace(xp):
            idctn = importlib.import_module("torch_dct").idct_2d
            del kwargs["axes"]
        else:
            msg = "idct is not implemented for this array namespace"
            raise NotImplementedError(msg)
    return idctn(x, **kwargs)


def _dst1(x, axis=-1):
    # O(2n log(2n)) implementation of dst type 1
    axis = normalize_axis_index(axis, x.ndim)
    shape = list(x.shape)
    shape[axis] = 1
    xp = array_namespace(x)
    zero = xp.broadcast_to(xp.zeros((1,), dtype=x.dtype), shape)
    x_tilde = xp.concat((zero, -x, zero, xp.flip(x, axis=axis)), axis=axis)
    x_tilde = xp.fft.rfft(x_tilde, axis=axis)
    slices = tuple(slice(1, -1) if i == axis else slice(None) for i in range(x.ndim))
    return xp.imag(x_tilde[*slices])


def _dstn1(x, type=1, axes=None):
    axes = (-1,) if axes is None else axes
    if type != 1:
        raise ValueError
    for axis in axes:
        x = _dst1(x, axis=axis)
    return x


def _idstn1(x, type=1, axes=None):
    xp = array_namespace(x)
    axes = (-1,) if axes is None else axes
    if type != 1:
        raise ValueError
    shape = x.shape
    sizes = []
    for axis in axes:
        sizes.append(shape[axis] + 1)
        x = _dst1(x, axis=axis)  # / (2 * size)  # ??? do one division at the end
    return idiv(x, (...,), (2 ** len(axes) * xp.prod(xp.asarray(sizes))))


def dst2(x, type=1, workers=None):
    # TODO(nin17): optional pyvkfft with cupy
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp):
        if __have_scipy__:
            spfft = importlib.import_module("scipy.fft")
            return spfft.dstn(x, type=type, axes=axes, workers=workers)
        msg = "Scipy required for the DST"
        raise NotImplementedError(msg)
    return _dstn1(x, type=type, axes=axes)


def idst2(x, type=1, workers=None):
    # TODO(nin17): optional pyvkfft with cupy
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp):
        if __have_scipy__:
            spfft = importlib.import_module("scipy.fft")
            return spfft.idstn(x, type=type, axes=axes, workers=workers)
        msg = "Scipy required for the DST"
        raise NotImplementedError(msg)
    return _idstn1(x, type=type, axes=axes)


if __have_numba__:
    import numpy as np
    from numba import extending, types
    from numba.core import errors

    @extending.overload(fft2)
    def fft2_overload(a, s=None, workers=None, use_rfft=True):
        axes = (-2, -1)
        if __have_scipy__:
            from scipy import fft as sp_fft

            if isinstance(use_rfft, types.NoneType):

                def impl(a, s=None, workers=None, use_rfft=True):
                    return sp_fft.fft2(a, s=s, axes=axes, workers=workers)

            else:

                def impl(a, s=None, workers=None, use_rfft=True):
                    return sp_fft.rfft2(a, s=s, axes=axes, workers=workers)

        elif isinstance(use_rfft, types.NoneType):

            def impl(a, s=None, workers=None, use_rfft=True):
                return np.fft.fft2(a, s=s, axes=axes)

        else:

            def impl(a, s=None, workers=None, use_rfft=True):
                return np.fft.rfft2(a, s=s, axes=axes)

        return impl

    @extending.overload(ifft2)
    def ifft2_overload(a, s=None, workers=None, use_rfft=True):

        axes = (-2, -1)
        if __have_scipy__:
            from scipy import fft as sp_fft

            def impl(a, s=None, workers=None, use_rfft=True):
                if use_rfft:
                    return sp_fft.irfft2(a, s=s, axes=axes, workers=workers)
                return sp_fft.ifft2(a, s=s, axes=axes, workers=workers).real

        else:

            def impl(a, s=None, workers=None, use_rfft=True):
                if use_rfft:
                    return np.fft.irfft2(a, s=s, axes=axes)
                return np.fft.ifft2(a, s=s, axes=axes).real

        return impl

    @extending.overload(dct2)
    def dct2_overload(x, workers=None):
        axes = (-2, -1)
        if __have_scipy__:
            from scipy import fft as spfft

            def impl(x, workers=None):
                return spfft.dctn(x, type=2, axes=axes, workers=workers)

            return impl
        msg = "Scipy is required for the DCT"
        raise errors.NumbaNotImplementedError(msg)

    @extending.overload(idct2)
    def idct2_overload(x, workers=None):
        axes = (-2, -1)
        if __have_scipy__:
            from scipy import fft as spfft

            def impl(x, workers=None):
                return spfft.idctn(x, type=2, axes=axes, workers=workers)

            return impl
        msg = "Scipy is required for the IDCT"
        raise errors.NumbaNotImplementedError(msg)

    @extending.overload(dst2)
    def dcs2_overload(x, type=1, workers=None):
        axes = (-2, -1)
        if __have_scipy__:
            from scipy import fft as spfft

            def impl(x, type=1, workers=None):
                return spfft.dstn(x, type=type, axes=axes, workers=workers)

            return impl
        msg = "Scipy is required for the DST"
        raise errors.NumbaNotImplementedError(msg)

    @extending.overload(idst2)
    def idst2_overload(x, type=1, workers=None):
        axes = (-2, -1)
        if __have_scipy__:
            from scipy import fft as spfft

            def impl(x, type=1, workers=None):
                return spfft.idstn(x, type=type, axes=axes, workers=workers)

            return impl
        msg = "Scipy is required for the IDST"
        raise errors.NumbaNotImplementedError(msg)
