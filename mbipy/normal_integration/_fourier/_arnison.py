"""Normal integration using the method of Arnison et al.

[Arnison, M. R., Larkin, K. G., Sheppard, C. J., Smith, N. I. & Cogswell, C. J.
Linear phase imaging using differential interference contrast microscopy.
Journal of microscopy 214, 7-12 (2004).](https://doi.org/10.1111/j.0022-2720.2004.01293.x)
"""

from __future__ import annotations

__all__ = ["arnison"]

from typing import TYPE_CHECKING, Literal

from mbipy.src.utils import array_namespace, astype, at, get_dtypes

from ._padding import antisymmetric
from ._utils import FFTMethod, fft_2d, ifft_2d, irfft_2d, rfft_2d

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

# TODO(nin17): dtype kwarg in np.fft.fftfreq/rfftfreq wait for numpy & rocketfft
# TODO(nin17): dtype promotion in numba


def arnison(
    gy: NDArray[floating],
    gx: NDArray[floating],
    pad: Literal["antisymmetric"] | None = None,
    workers: int | None = None,
    fft_method: FFTMethod = FFTMethod.FFT,
) -> NDArray[floating]:
    """Perform normal integration using the method of Arnison et al[^1].

    ??? info "Array API Compatibility"

        {{ NormalIntegration.row("arnison") | indent(4) }}

    !!! example "[Example][arnison-example]"

    [^1]:[Arnison, M. R., Larkin, K. G., Sheppard, C. J., Smith, N. I. & Cogswell, C. J.
    Linear phase imaging using differential interference contrast microscopy.
    Journal of microscopy 214, 7-12 (2004).](https://doi.org/10.1111/j.0022-2720.2004.01293.x)

    Parameters
    ----------
    gy : NDArray[floating] (..., M, N)
        Vertical gradient(s).
    gx : NDArray[floating] (..., M, N)
        Horizontal gradient(s).
    pad : Literal["antisymmetric"] | None, optional
        Type of padding to apply:
        ["antisymmetric"][mbipy.normal_integration.padding.antisymmetric] | None,
        by default `None`
    workers : int | None, optional
        Passed to [scipy.fft.fft2][]/[scipy.fft.rfft2][] &
        [scipy.fft.ifft2][]/[scipy.fft.irfft2][] if `gy` & `gx` are numpy
        arrays and
        [config.use_scipy_fft][mbipy.src.config.Config.use_scipy_fft] = True,
        by default `None`
    fft_method : FFTMethod, optional
        FFT method to use, by default FFTMethod.FFT


    Returns
    -------
    NDArray[floating] (..., M, N)
        Normal field(s).

    Raises
    ------
    ValueError
        If the input arrays are not real-valued.
    ValueError
        If the pad argument is not None or "antisymmetric".
    """
    xp = array_namespace(gy, gx)
    dtype, cdtype = get_dtypes(gy, gx)
    y, x = xp.broadcast_shapes(gx.shape, gy.shape)[-2:]
    y2, x2 = 2 * y if pad else y, 2 * x if pad else x

    match pad:
        case "antisymmetric":
            gy, gx = antisymmetric(gy=gy, gx=gx)
        case None:
            ...
        case _:
            msg = f"Invalid value for pad: {pad}"
            raise ValueError(msg)

    match fft_method:
        case FFTMethod.FFT:
            fx = xp.fft.fftfreq(x2)  # FIXME: dtype kwarg
            gyc = astype(gy, cdtype, copy=True)  # FIXME: gyc = gy * 1.0j
            gyc = at(gyc)[:].multiply(1.0j)
            operand = gx + gyc
            f_num = fft_2d(operand, workers=workers)
        case FFTMethod.RFFT:
            fx = xp.fft.rfftfreq(x2)  # FIXME: dtype kwarg
            s = (y2, x2)
            gxfft2 = rfft_2d(gx, s=s, workers=workers)
            gyfft2 = rfft_2d(gy, s=s, workers=workers)
            gyfft2 = at(gyfft2)[:].multiply(1.0j)
            f_num = gxfft2 + gyfft2
        case _:
            msg = f"Invalid value for fft_method: {fft_method}"
            raise ValueError(msg)

    fx = astype(fx, dtype)  # FIXME: dtype kwarg
    fy = astype(xp.fft.fftfreq(y2), dtype)  # FIXME: dtype kwarg
    fx = at(fx)[:].multiply(xp.pi)
    fy = at(fy)[:].multiply(xp.pi)

    sinfx = astype(xp.sin(fx), cdtype)  # FIXME: sinfx = xp.sin(fx) * 2.0j
    sinfy = xp.sin(fy)[:, None]
    sinfx = at(sinfx)[:].multiply(2.0j)
    sinfy = at(sinfy)[:].multiply(-2.0)

    f_den = sinfx + sinfy
    f_den = at(f_den)[..., 0, 0].set(1.0)  # avoid division by zero warning
    f_num = at(f_num)[:].divide(f_den)
    f_num = at(f_num)[..., 0, 0].set(0.0)

    match fft_method:
        case FFTMethod.FFT:
            return xp.real(ifft_2d(f_num, workers=workers)[..., :y, :x])
        case FFTMethod.RFFT:
            return irfft_2d(f_num, s=s, workers=workers)[..., :y, :x]
        case _:
            msg = f"Invalid value for fft_method: {fft_method}"
            raise ValueError(msg)
