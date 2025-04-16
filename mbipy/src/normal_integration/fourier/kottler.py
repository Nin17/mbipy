"""Normal integration using the method of Kottler et al.

Kottler, C., David, C., Pfeiffer, F. & Bunk, O. A two-directional approach for
grating based differential phase contrast imaging using hard x-rays 2007.
"""

from __future__ import annotations

__all__ = ("kottler",)


from typing import TYPE_CHECKING

from mbipy.src.normal_integration.fourier.utils import fft2, ifft2
from mbipy.src.normal_integration.padding import antisym
from mbipy.src.normal_integration.utils import check_shapes
from mbipy.src.utils import array_namespace, cast_scalar, idiv, setitem

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


def kottler(
    gy: NDArray[floating],
    gx: NDArray[floating],
    pad: str | None = None,
    workers: int | None = None,
    use_rfft: bool = True,
) -> NDArray[floating]:
    """Perform normal integration using the method of Kottler et al.

    Kottler, C., David, C., Pfeiffer, F. & Bunk, O.
    A two-directional approach for grating based differential phase contrast imaging
    using hard x-rays 2007.

    Parameters
    ----------
    gy : (..., M, N) NDArray[floating]
        Vertical gradient(s).
    gx : (..., M, N) NDArray[floating]
        Horizontal gradient(s).
    pad : str | None, optional
        Type of padding to apply: "antisym" | None , by default None
    workers : int | None, optional
        Passed to scipy.fft fftn & ifftn, by default None
    use_rfft : bool, optional
        Use a rfftn instead of fftn, by default True

    Returns
    -------
    (..., M, N) NDArray[floating]
        Normal field(s).

    Raises
    ------
    ValueError
        If the input arrays are not real-valued.
    ValueError
        If the pad argument is not None or "antisym".

    """
    xp = array_namespace(gy, gx)
    dtype = xp.result_type(gy, gx)
    cdtype = xp.result_type(dtype, xp.complex64)
    if not xp.isdtype(dtype, "real floating"):
        msg = "Input arrays must be real-valued."
        raise ValueError(msg)
    y, x = check_shapes(gx, gy)
    y2, x2 = 2 * y if pad else y, 2 * x if pad else x
    s = (y2, x2)

    if pad == "antisym":
        gy, gx = antisym(gy=gy, gx=gx)
    elif pad is None:
        pass
    else:
        msg = f"Invalid value for pad: {pad}"
        raise ValueError(msg)

    # !!! Cast scalars to the same dtype as the result. Necessary for Numba.
    zero = cast_scalar(0.0, dtype)
    one = cast_scalar(1.0, dtype)
    two = cast_scalar(2.0, dtype)
    one_j = cast_scalar(1j, cdtype)

    if use_rfft:
        fx = xp.astype(xp.fft.rfftfreq(x2), dtype, copy=False)
    else:
        fx = xp.astype(xp.fft.fftfreq(x2), dtype, copy=False)
    fy = xp.astype(xp.fft.fftfreq(y2)[:, None], dtype, copy=False)
    if use_rfft:
        gxfft = fft2(gx, s=s, workers=workers, use_rfft=use_rfft)
        gyfft = fft2(gy, s=s, workers=workers, use_rfft=use_rfft)
        f_num = gxfft + one_j * gyfft
    else:
        f_num = fft2(gx + one_j * gy, s=s, workers=workers, use_rfft=None)
    f_den = one_j * two * xp.pi * (fx + one_j * fy)

    f_den = setitem(f_den, (..., 0, 0), one)  # avoid division by zero warning
    f_phase = idiv(f_num, (...,), f_den)  # f_num is f_phase
    f_phase = setitem(f_phase, (..., 0, 0), zero)

    return xp.real(ifft2(f_phase, s=s, workers=workers, use_rfft=use_rfft)[..., :y, :x])
