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

from mbipy.src.config import __have_scipy__
from mbipy.src.utils import array_namespace, idiv

if TYPE_CHECKING:

    from numpy import complexfloating, floating
    from numpy.typing import NDArray


# TODO(nin17): optional pyvkfft with cupy


def rfft_2d(
    a: NDArray[floating],
    s: tuple[int, int],
    workers: int | None = None,
) -> NDArray[complexfloating]:
    """Compute the FFT of the last two axes of a real-valued array.

    Parameters
    ----------
    a : NDArray[floating]
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
    xp = array_namespace(a)
    if is_numpy_namespace(xp) and __have_scipy__:
        fft = importlib.import_module("scipy.fft")
        return fft.rfft2(a, s=s, axes=axes, workers=workers)
    return xp.fft.rfftn(a, s=s, axes=axes)


def irfft_2d(
    a: NDArray[complexfloating],
    s: tuple[int, int],
    workers: int | None = None,
) -> NDArray[floating]:
    """Compute the inverse FFT of the last two axes of a complex-valued array.

    Parameters
    ----------
    a : NDArray[complexfloating]
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
    xp = array_namespace(a)
    if is_numpy_namespace(xp) and __have_scipy__:
        fft = importlib.import_module("scipy.fft")
        return fft.irfft2(a, s=s, axes=axes, workers=workers)
    return xp.fft.irfftn(a, s=s, axes=axes)


def fft_2d(
    a: NDArray[complexfloating],
    workers: int | None = None,
) -> NDArray[complexfloating]:
    """Compute the FFT of the last two axes of a complex-valued array.

    Parameters
    ----------
    a : NDArray[complexfloating]
        Input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[complexfloating]
        Transformed input array.
    """
    axes = (-2, -1)
    xp = array_namespace(a)
    if is_numpy_namespace(xp) and __have_scipy__:
        _fft = importlib.import_module("scipy.fft")
        return _fft.fft2(a, axes=axes, workers=workers)
    return xp.fft.fftn(a, axes=axes)


def ifft_2d(
    a: NDArray[complexfloating],
    workers: int | None = None,
) -> NDArray[complexfloating]:
    """Compute the inverse FFT of the last two axes of a complex-valued array.

    Parameters
    ----------
    a : NDArray[complexfloating]
        Input array.
    workers : int | None, optional
        Maximum number of parallel workers (used by SciPy), by default None

    Returns
    -------
    NDArray[complexfloating]
        Transformed input array.
    """
    axes = (-2, -1)
    xp = array_namespace(a)
    if is_numpy_namespace(xp) and __have_scipy__:
        fft = importlib.import_module("scipy.fft")
        return fft.ifft2(a, axes=axes, workers=workers)
    return xp.fft.ifftn(a, axes=axes)


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
        _description_
    NotImplementedError
        _description_
    """
    # TODO(nin17): docstring
    # ??? could implement this myself like the DST - remove torch_dct dependency
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


def idct2_2d(x: NDArray[floating], workers: int | None = None) -> NDArray[floating]:
    """Type II IDCT along the last two axes, norm="backward".

    Parameters
    ----------
    x : NDArray[floating]
        _description_
    workers : int | None, optional
        _description_, by default None

    Returns
    -------
    NDArray[floating]
        _description_

    Raises
    ------
    ImportError
        _description_
    NotImplementedError
        _description_
    """
    # TODO(nin17): docstring
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


def _dst1(x: NDArray[floating], axis: int = -1) -> NDArray[floating]:
    """Compute the Type I DST along the given axis. O(2n log(2n)).

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
        if __have_scipy__:
            spfft = importlib.import_module("scipy.fft")
            return spfft.dstn(x, type=1, axes=axes, workers=workers)
        msg = "x is a numpy array and scipy isn't installed - fallback to slow method."
        warnings.warn(msg, stacklevel=2)
    return _dst1_nd(x, axes=axes)


def idst1_2d(
    x: NDArray[floating],
    workers: int | None = None,
) -> NDArray[floating]:
    """Compute the Type I IDST along the last two axes.

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
        if __have_scipy__:
            spfft = importlib.import_module("scipy.fft")
            return spfft.idstn(x, type=1, axes=axes, workers=workers)
        msg = "x is a numpy array and scipy isn't installed - fallback to slow method."
        warnings.warn(msg, stacklevel=2)
    return _idst1_nd(x, axes=axes)
