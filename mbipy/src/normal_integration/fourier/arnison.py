"""Normal integration using the method of Arnison et al.

Arnison, M. R., Larkin, K. G., Sheppard, C. J., Smith, N. I. & Cogswell, C. J.
Linear phase imaging using differential interference contrast microscopy.
Journal of microscopy 214, 7-12 (2004).
"""

from __future__ import annotations

__all__ = ("arnison",)

from typing import TYPE_CHECKING, Literal

from mbipy.src.normal_integration.fourier.padding import antisymmetric
from mbipy.src.normal_integration.fourier.utils import (
    FFTMethod,
    fft_2d,
    ifft_2d,
    irfft_2d,
    rfft_2d,
)
from mbipy.src.normal_integration.utils import check_shapes
from mbipy.src.utils import (
    array_namespace,
    astype,
    get_dtypes,
    idiv,
    imul,
    setitem,
)

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


def arnison(
    gy: NDArray[floating],
    gx: NDArray[floating],
    pad: Literal["antisymmetric"] | None = None,
    workers: int | None = None,
    fft_method: FFTMethod = FFTMethod.FFT,
) -> NDArray[floating]:
    """Perform normal integration using the method of Arnison et al.

    Arnison, M. R., Larkin, K. G., Sheppard, C. J., Smith, N. I. & Cogswell, C. J.
    Linear phase imaging using differential interference contrast microscopy.
    Journal of microscopy 214, 7-12 (2004).

    Parameters
    ----------
    gy : (..., M, N) NDArray[floating]
        Vertical gradient(s).
    gx : (..., M, N) NDArray[floating]
        Horizontal gradient(s).
    pad : Literal["antisymmetric"] | None, optional
        Type of padding to apply: "antisymmetric" | None, by default None
    workers : int | None, optional
        Passed to scipy.fft fft2 & ifft2, by default None
    fft_method : FFTMethod, optional
        FFT method to use, by default FFTMethod.FFT


    Returns
    -------
    (..., M, N) NDArray[floating]
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
    y, x = check_shapes(gx, gy)
    y2, x2 = 2 * y if pad else y, 2 * x if pad else x

    if pad == "antisymmetric":
        gy, gx = antisymmetric(gy=gy, gx=gx)
    elif pad is None:
        pass
    else:
        msg = f"Invalid value for pad: {pad}"
        raise ValueError(msg)

    fx = astype(
        xp.fft.rfftfreq(x2) if fft_method == FFTMethod.RFFT else xp.fft.fftfreq(x2),
        dtype,
    )
    fy = astype(xp.fft.fftfreq(y2), dtype)
    fx = imul(fx, ..., xp.pi)
    fy = imul(fy, ..., xp.pi)

    sinfx = astype(xp.sin(fx), cdtype)  # ??? do inplace
    sinfy = xp.sin(fy)[:, None]
    sinfx = imul(sinfx, ..., 2.0j)
    sinfy = imul(sinfy, ..., -2.0)

    if fft_method == FFTMethod.RFFT:
        s = (y2, x2)
        gxfft2 = rfft_2d(gx, s=s, workers=workers)
        gyfft2 = rfft_2d(gy, s=s, workers=workers)
        gyfft2 = imul(gyfft2, ..., 1.0j)
        f_num = gxfft2 + gyfft2
    else:
        gyc = astype(gy, cdtype, copy=True)
        gyc = imul(gyc, ..., 1.0j)
        operand = gx + gyc
        f_num = fft_2d(operand, workers=workers)

    f_den = sinfx + sinfy
    f_den = setitem(f_den, (..., 0, 0), 1.0)  # avoid division by zero warning
    frac = idiv(f_num, ..., f_den)
    frac = setitem(frac, (..., 0, 0), 0.0)
    if fft_method == FFTMethod.RFFT:
        return irfft_2d(frac, s=s, workers=workers)[..., :y, :x]
    return xp.real(ifft_2d(frac, workers=workers)[..., :y, :x])
