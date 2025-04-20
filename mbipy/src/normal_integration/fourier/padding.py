"""Antisymmetric padding of the input gradients."""

from __future__ import annotations

__all__ = ("antisym",)

from typing import TYPE_CHECKING

from mbipy.src.normal_integration.utils import check_shapes
from mbipy.src.utils import array_namespace, setitem

if TYPE_CHECKING:

    from numpy import floating
    from numpy.typing import NDArray


def flip(a: NDArray, axis: int | tuple[int, ...] | None = None) -> NDArray:
    # !!! To avoid numba overload of numpy function
    xp = array_namespace(a)
    return xp.flip(a, axis=axis)


def antisym(
    gy: NDArray[floating],
    gx: NDArray[floating],
) -> tuple[NDArray[floating], NDArray[floating]]:
    """Antisymmetric padding of the input gradients.

    Parameters
    ----------
    gy : (..., M, N) NDArray[floating]
        Vertical gradient(s).
    gx : (..., M, N) NDArray[floating]
        Horizontal gradient(s).

    Returns
    -------
    tuple[(..., 2M, 2N) NDArray[floating], (..., 2M, 2N) NDArray[floating]]
        Vertical and horizontal gradients with antisymmetric padding.

    """
    xp = array_namespace(gy, gx)
    y, x = check_shapes(gy, gx)

    gx_neg = -gx
    gy_neg = -gy

    _sy = slice(y)
    _sx = slice(x)
    sy_ = slice(y, None)
    sx_ = slice(x, None)

    as_gx = xp.empty(gx.shape[:-2] + (y * 2, x * 2), dtype=gx.dtype)
    as_gx = setitem(as_gx, (..., _sy, _sx), gx)
    as_gx = setitem(as_gx, (..., sy_, _sx), flip(gx, axis=-2))
    as_gx = setitem(as_gx, (..., _sy, sx_), flip(gx_neg, axis=-1))
    as_gx = setitem(as_gx, (..., sy_, sx_), flip(gx_neg, axis=(-2, -1)))
    # Equivalent to:
    # as_gx[..., :y, :x] = gx
    # as_gx[..., y:, :x] = gx[..., ::-1, :]
    # as_gx[..., :y, x:] = gx_neg[..., ::-1]
    # as_gx[..., y:, x:] = gx_neg[..., ::-1, ::-1]

    as_gy = xp.empty(gy.shape[:-2] + (y * 2, x * 2), dtype=gy.dtype)
    as_gy = setitem(as_gy, (..., _sy, _sx), gy)
    as_gy = setitem(as_gy, (..., sy_, _sx), flip(gy_neg, axis=-2))
    as_gy = setitem(as_gy, (..., _sy, sx_), flip(gy, axis=-1))
    as_gy = setitem(as_gy, (..., sy_, sx_), flip(gy_neg, axis=(-2, -1)))
    # Equivalent to:
    # as_gy[..., :y, :x] = gy
    # as_gy[..., y:, :x] = gy_neg[..., ::-1, :]
    # as_gy[..., :y, x:] = gy[..., ::-1]
    # as_gy[..., y:, x:] = gy_neg[..., ::-1, ::-1]

    return as_gy, as_gx
