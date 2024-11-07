"""Normal integration using the method of Arnison et al.

Arnison, M. R., Larkin, K. G., Sheppard, C. J., Smith, N. I. & Cogswell, C. J.
Linear phase imaging using differential interference contrast microscopy.
Journal of microscopy 214, 7-12 (2004).
"""

from __future__ import annotations

__all__ = ("arnison",)

from typing import TYPE_CHECKING, overload

from mbipy.src.config import __have_jax__
from mbipy.src.normal_integration.padding import antisym
from mbipy.src.normal_integration.utils import check_shapes, get_dfts
from mbipy.src.utils import array_namespace, setitem

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

    @overload
    def arnison(
        gy: NDArray[floating],
        gx: NDArray[floating],
        pad: str | None = None,
        workers: int | None = None,
    ) -> NDArray[floating]: ...

    try:
        from jax import Array

        @overload
        def arnison(
            gy: Array,
            gx: Array,
            pad: str | None = None,
            workers: int | None = None,
        ) -> Array: ...

    except ImportError:
        pass


def arnison(gy, gx, pad=None, workers=None):
    # TODO(nin17): docstring
    xp = array_namespace(gy, gx)
    dtype = xp.result_type(gy, gx)
    if not xp.isdtype(dtype, "real floating"):
        msg = "Input arrays must be real-valued."
        raise ValueError(msg)
    fft2, ifft2 = get_dfts(xp)
    y, x = check_shapes(gx, gy)

    if pad == "antisym":
        gy, gx = antisym(gy=gy, gx=gx)
    elif pad is None:
        pass
    else:
        msg = f"Invalid value for pad: {pad}"
        raise ValueError(msg)

    fx = xp.astype(xp.fft.fftfreq(2 * x if pad else x), dtype, copy=False)
    fy = xp.astype(xp.fft.fftfreq(2 * y if pad else y)[:, None], dtype, copy=False)

    sinfx = xp.sin(xp.pi * fx)
    sinfy = xp.sin(xp.pi * fy)

    f_num = fft2(gx + 1j * gy, axes=(-2, -1), workers=workers)
    f_den = 2j * (sinfx + 1j * sinfy)
    # avoid division by zero warning
    f_den = setitem(f_den, (..., 0, 0), 1.0, xp)
    frac = f_num / f_den
    frac = setitem(frac, (..., 0, 0), 0.0, xp)
    return ifft2(frac, axes=(-2, -1), workers=workers).real[..., :y, :x]
