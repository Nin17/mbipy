"""Utilities for normal integration methods that use DFT, DCT or DST transforms."""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING

from array_api_compat import (
    is_cupy_namespace,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)
from numpy.lib.array_utils import normalize_axis_index

from mbipy.src.config import config as cfg
from mbipy.src.utils import array_namespace, idiv

if TYPE_CHECKING:  # pragma: no cover
    from numpy import complexfloating, floating
    from numpy.typing import NDArray



def _check_s(s):
    if not len(s) == 2:
        msg = "s must be a tuple of length 2"
        raise ValueError(msg)
    if not all(isinstance(i, int) for i in s):
        msg = "s must be a tuple of integers"
        raise ValueError(msg)

def _contiguous(x: NDArray) -> NDArray:
    xp = array_namespace(x)
    if not (x.flags.c_contiguous or x.flags.f_contiguous):
        x = xp.ascontiguousarray(x)
    return x


def rfft_2d(
    x: NDArray[floating],
    s: tuple[int, int],
    workers: int | None = None,
) -> NDArray[complexfloating]:
    """Compute the FFT of the last two axes of a real-valued array.

    Parameters
    ----------
    x : NDArray[floating]
        Input array.
    s : tuple[int, int]
        Length along last two axes to use from input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[complexfloating]
        Transformed input array.
    """
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp) and cfg.have_scipy:
        fft = importlib.import_module("scipy.fft")
        return fft.rfft2(x, s=s, axes=axes, workers=workers)
    if is_cupy_namespace(xp) and cfg.use_pyvkfft:
        fft = importlib.import_module("pyvkfft.fft")
        _check_s(s)
        if s != x.shape[-2:]:
            msg = "s must be equal to the last two dimensions of x"
            raise ValueError(msg)
        x = _contiguous(x)
        return fft.rfftn(x, ndim=2)
    return xp.fft.rfftn(x, s=s, axes=axes)


def irfft_2d(
    x: NDArray[complexfloating],
    s: tuple[int, int],
    workers: int | None = None,
) -> NDArray[floating]:
    """Compute the inverse FFT of the last two axes of a complex-valued array.

    Parameters
    ----------
    x : NDArray[complexfloating]
        Input array.
    s : tuple[int, int]
        Length along last two axes to use from input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[floating]
        Transformed input array.
    """
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp) and cfg.have_scipy:
        fft = importlib.import_module("scipy.fft")
        return fft.irfft2(x, s=s, axes=axes, workers=workers)
    if is_cupy_namespace(xp) and cfg.use_pyvkfft:
        fft = importlib.import_module("pyvkfft.fft")
        _check_s(s)
        if s != x.shape[-2:]:
            if not all(i >= j for i, j in zip(s, x.shape[-2:])):
                msg = "s must be greater than or equal to the last two dimensions of x"
                raise ValueError(msg)
            shape = x.shape[:-2] + s
            _x = xp.zeros(shape, dtype=x.dtype)
            _x[..., : x.shape[-2], : x.shape[-1]] = x
            x = _x
        else:
            x = _contiguous(x)
        return fft.irfftn(x, ndim=2)
    return xp.fft.irfftn(x, s=s, axes=axes)


def fft_2d(
    x: NDArray[complexfloating],
    workers: int | None = None,
) -> NDArray[complexfloating]:
    """Compute the FFT of the last two axes of a complex-valued array.

    Parameters
    ----------
    x : NDArray[complexfloating]
        Input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[complexfloating]
        Transformed input array.
    """
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp) and cfg.have_scipy:
        fft = importlib.import_module("scipy.fft")
        return fft.fft2(x, axes=axes, workers=workers)
    if is_cupy_namespace(xp) and cfg.use_pyvkfft:
        fft = importlib.import_module("pyvkfft.fft")
        x = _contiguous(x)
        if not xp.isdtype(x, "complex floating"):
            msg = "x must be a complex-valued array"
            raise ValueError(msg)
        return fft.fftn(x, ndim=2)
    return xp.fft.fftn(x, axes=axes)


def ifft_2d(
    x: NDArray[complexfloating],
    workers: int | None = None,
) -> NDArray[complexfloating]:
    """Compute the inverse FFT of the last two axes of a complex-valued array.

    Parameters
    ----------
    x : NDArray[complexfloating]
        Input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[complexfloating]
        Transformed input array.
    """
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp) and cfg.have_scipy:
        fft = importlib.import_module("scipy.fft")
        return fft.ifft2(x, axes=axes, workers=workers)
    if is_cupy_namespace(xp) and cfg.use_pyvkfft:
        fft = importlib.import_module("pyvkfft.fft")
        x = _contiguous(x)
        if not xp.isdtype(x, "complex floating"):
            msg = "x must be a complex-valued array"
            raise ValueError(msg)
            
        return fft.ifftn(x, ndim=2)
    return xp.fft.ifftn(x, axes=axes)


def dct2_2d(x: NDArray[floating], workers: int | None = None) -> NDArray[floating]:
    """Type II DCT along the last two axes, norm="backward".

    Parameters
    ----------
    x : NDArray[floating]
        Input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[floating]
        Transformed input array.

    Raises
    ------
    ImportError
        If the array namespace is numpy and scipy is not installed.
        If the array namespace is torch and torch-dct is not installed.
    NotImplementedError
        If the array namespace is not cupy, jax, numpy or torch.
    """
    # ??? could implement this myself like the DST - remove torch_dct dependency
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp):
        if not cfg.have_scipy:
            msg = "Need SciPy for the DCT"
            raise ImportError(msg)
        fft = importlib.import_module("scipy.fft")
        return fft.dctn(x, type=2, axes=axes, workers=workers)
    elif is_cupy_namespace(xp):
        if cfg.use_pyvkfft:
            fft = importlib.import_module("pyvkfft.fft")
            x = _contiguous(x)
            return fft.dctn(x, ndim=2, dct_type=2)
        fft = importlib.import_module("cupyx.scipy.fft")
        return fft.dctn(x, type=2, axes=axes)
    elif is_jax_namespace(xp):
        fft = importlib.import_module("jax.scipy.fft")
        return fft.dctn(x, type=2, axes=axes)
    elif is_torch_namespace(xp):
        dctn = importlib.import_module("torch_dct").dct_2d
        return dctn(x)
    else:
        msg = "dct is not implemented for this array namespace."
        raise NotImplementedError(msg)


def idct2_2d(x: NDArray[floating], workers: int | None = None) -> NDArray[floating]:
    """Type II IDCT along the last two axes, norm="backward".

    Parameters
    ----------
    x : NDArray[floating]
        Input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[floating]
        Transformed input array.

    Raises
    ------
    ImportError
        If the array namespace is numpy and scipy is not installed.
        If the array namespace is torch and torch-dct is not installed.
    NotImplementedError
        If the array namespace is not cupy, jax, numpy or torch.
    """
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp):
        if not cfg.have_scipy:
            msg = "Need SciPy for the IDCT"
            raise ImportError(msg)
        fft = importlib.import_module("scipy.fft")
        return fft.idctn(x, type=2, axes=axes, workers=workers)
    elif is_cupy_namespace(xp):
        if cfg.use_pyvkfft:
            fft = importlib.import_module("pyvkfft.fft")
            x = _contiguous(x)
            return fft.idctn(x, ndim=2, dct_type=2)
        fft = importlib.import_module("cupyx.scipy.fft")
        return fft.idctn(x, type=2, axes=axes)
    elif is_jax_namespace(xp):
        fft = importlib.import_module("jax.scipy.fft")
        return fft.idctn(x, type=2, axes=axes)
    elif is_torch_namespace(xp):
        idctn = importlib.import_module("torch_dct").idct_2d
        return idctn(x)
    else:
        msg = "idct is not implemented for this array namespace"
        raise NotImplementedError(msg)


def _dst1(x: NDArray[floating], axis: int = -1) -> NDArray[floating]:
    """Compute the Type I DST along the given axis. O(2n log(2n)).

    norm="backward", orthogonalize=False

    Parameters
    ----------
    x : NDArray[floating]
        Input array.
    axis : int, optional
        Axis along which the Type I DST is applied, by default -1

    Returns
    -------
    NDArray[floating]
        Transformed input array.
    """
    axis = normalize_axis_index(axis, x.ndim)
    shape = list(x.shape)
    shape[axis] = 1
    xp = array_namespace(x)
    zero = xp.broadcast_to(xp.zeros((1,), dtype=x.dtype), shape)
    x_tilde = xp.concat((zero, -x, zero, xp.flip(x, axis=axis)), axis=axis)
    x_tilde = xp.fft.rfft(x_tilde, axis=axis)
    slices = tuple(slice(1, -1) if i == axis else slice(None) for i in range(x.ndim))
    return xp.imag(x_tilde[slices])


def _dst1_nd(
    x: NDArray[floating],
    axes: tuple[int, ...] | None = None,
) -> NDArray[floating]:
    """Compute the Type I DST along the given axes. O(2n log(2n)).

    norm="backward", orthogonalize=False

    Parameters
    ----------
    x : NDArray[floating]
        Input array.
    axes : tuple[int, ...] | None, optional
        Axes over which to apply the Type I DST, by default None

    Returns
    -------
    NDArray[floating]
        Transformed input array.
    """
    axes = (-1,) if axes is None else axes
    for axis in axes:
        x = _dst1(x, axis=axis)
    return x


def _idst1_nd(
    x: NDArray[floating],
    axes: tuple[int, ...] | None = None,
) -> NDArray[floating]:
    """Compute the Type I IDST along the given axes. O(2n log(2n)).

    norm="backward", orthogonalize=False

    Parameters
    ----------
    x : NDArray[floating]
        Input array.
    axes : tuple[int, ...] | None, optional
        Axes over which to compute the Type I IDST, by default None

    Returns
    -------
    NDArray[floating]
        Transformed input array.
    """
    xp = array_namespace(x)
    axes = (-1,) if axes is None else axes
    shape = x.shape
    sizes = []
    for axis in axes:
        sizes.append(shape[axis] + 1)
        x = _dst1(x, axis=axis)  # / (2 * size)  # ??? do one division at the end
    return idiv(x, (...,), (2 ** len(axes) * xp.prod(xp.asarray(sizes))))


def dst1_2d(
    x: NDArray[floating],
    workers: int | None = None,
) -> NDArray[floating]:
    """Compute the Type I DST along the last two axes.

    norm="backward", orthogonalize=False

    Parameters
    ----------
    x : NDArray[floating]
        Input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[floating]
        Transformed input array.
    """
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp):
        if cfg.have_scipy:
            fft = importlib.import_module("scipy.fft")
            return fft.dstn(x, type=1, axes=axes, workers=workers)
        msg = "x is a numpy array and scipy isn't installed - fallback to slow method."
        warnings.warn(msg, stacklevel=2)
    if is_cupy_namespace(xp) and cfg.have_pyvkfft:
        fft = importlib.import_module("pyvkfft.fft")
        x = _contiguous(x)
        return fft.dstn(x, ndim=2, dst_type=1)
    return _dst1_nd(x, axes=axes)


def idst1_2d(
    x: NDArray[floating],
    workers: int | None = None,
) -> NDArray[floating]:
    """Compute the Type I IDST along the last two axes.

    norm="backward", orthogonalize=False

    Parameters
    ----------
    x : NDArray[floating]
        Input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[floating]
        Transformed input array.
    """
    axes = (-2, -1)
    xp = array_namespace(x)
    if is_numpy_namespace(xp):
        if cfg.have_scipy:
            fft = importlib.import_module("scipy.fft")
            return fft.idstn(x, type=1, axes=axes, workers=workers)
        msg = "x is a numpy array and scipy isn't installed - fallback to slow method."
        warnings.warn(msg, stacklevel=2)
    if is_cupy_namespace(xp) and cfg.have_pyvkfft:
        fft = importlib.import_module("pyvkfft.fft")
        x = _contiguous(x)
        return fft.idstn(x, ndim=2, dst_type=1)
    return _idst1_nd(x, axes=axes)
