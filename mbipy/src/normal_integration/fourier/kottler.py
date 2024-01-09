"""_summary_
"""
import types
import typing
import warnings

import numpy as np
from numpy.typing import NDArray


def create_kottler(
    xp: types.ModuleType, antisym_pad: typing.Callable
) -> typing.Callable:
    docstring = """
    Integrate the normal vector field to obtain the scalar field using the
    Kottler method with antisymmetric padding.
    # TODO nin17: reference the paper

    Parameters
    ----------
    gx : ArrayLike
        (..., M, N)
        The horizontal component of the normal vector field. Horizontal
        gradient of the scalar field.
    gy : ArrayLike
        (..., M, N)
        The vertical component of the normal vector field. Vertical gradient of
        the scalar field.

    Returns
    -------
    Array
        (..., M, N)
        The scalar field
    """

    if xp.__name__ == "jax.numpy":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fdtype = xp.array(0.0, dtype=np.float64).dtype

        def kottler(*, gy, gx, pad: bool = True):
            assert gx.shape[-2:] == gy.shape[-2:]
            assert gx.ndim >= 2
            y, x = gx.shape[-2:]

            if pad:
                gy, gx = antisym_pad(gy=gy, gx=gx)

            gx_fft = xp.fft.fft2(gx, axes=(-2, -1))
            gy_fft = xp.fft.fft2(gy, axes=(-2, -1))

            fx = xp.fft.fftfreq(gx.shape[-1]).reshape((1,) * (gx.ndim - 1) + (-1,))
            fy = xp.fft.fftfreq(gy.shape[-2]).reshape((1,) * (gx.ndim - 2) + (-1, 1))

            f_num = gx_fft + 1j * gy_fft
            f_den = 1j * 2 * xp.pi * (fx + 1j * fy) + xp.finfo(fdtype).eps
            f_phase = f_num / f_den

            f_phase = f_phase.at[..., 0, 0].set(0.0 + 0.0j)

            return xp.fft.ifft2(f_phase, axes=(-2, -1)).real[..., :y, :x]

    else:

        def kottler(*, gy, gx, pad: bool = True):
            assert gx.shape[-2:] == gy.shape[-2:]
            assert gx.ndim >= 2

            y, x = gx.shape[-2:]

            if pad:
                gy, gx = antisym_pad(gy=gy, gx=gx)

            gx_fft = xp.fft.fft2(gx, axes=(-2, -1))
            gy_fft = xp.fft.fft2(gy, axes=(-2, -1))

            fx = xp.fft.fftfreq(gx.shape[-1]).reshape((1,) * (gx.ndim - 1) + (-1,))
            fy = xp.fft.fftfreq(gy.shape[-2]).reshape((1,) * (gx.ndim - 2) + (-1, 1))

            f_num = gx_fft + 1j * gy_fft
            f_den = 1j * 2 * xp.pi * (fx + 1j * fy) + xp.finfo(xp.float64).eps
            f_phase = f_num / f_den

            f_phase[..., 0, 0] = 0.0 + 0.0j

            return xp.fft.ifft2(f_phase, axes=(-2, -1)).real[..., :y, :x]

    return kottler
