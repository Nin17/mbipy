"""Antisymmetric padding of the input gradients."""

from __future__ import annotations

__all__ = ["antisymmetric"]

from typing import TYPE_CHECKING

from mbipy.src.utils import array_namespace, setitem

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


# ??? redo padding do it edge value - values like in numpy rather than just - values
# ??? possibly better for large values etc...
# TODO(nin17): use xpx.at


def flip(a: NDArray, axis: int | tuple[int, ...] | None = None) -> NDArray:
    """Reverses the order of elements in an array along the given axis.

    Parameters
    ----------
    a : NDArray
        Input array.
    axis : int | tuple[int, ...] | None, optional
        The axis or axes along which to flip. If `axis` is `None`, flip along all axes,
        by default `None`

    Returns
    -------
    NDArray
        Input array with the order of elements reversed along the specified axes.
    """
    # Need to use `flip` for pytorch as negative steps are not supported in indexing
    # Use a wrapper for `flip` to avoid overload of numpy function
    xp = array_namespace(a)
    return xp.flip(a, axis=axis)


def antisymmetric(
    gy: NDArray[floating],
    gx: NDArray[floating],
) -> tuple[NDArray[floating], NDArray[floating]]:
    """Antisymmetric padding of the input gradients[^1].

    ??? info "Array API Compatibility"

        {{ NormalIntegrationPadding.row("antisymmetric") | indent(4) }}

    !!! example "[Example][antisymmetric-example]"

    [^1]:[P. Bon, S. Monneret, and B. Wattellier, “Noniterative boundary-artifact-free
    wavefront reconstruction from its derivatives,” Appl. Opt., vol. 51, no. 23,
    pp. 5698-5704, Aug. 2012. DOI: 10.1364/AO.51.005698.](https://doi.org/10.1364/AO.51.005698)

    Parameters
    ----------
    gy : NDArray[floating] (..., M, N)
        Vertical gradient(s).
    gx : NDArray[floating] (..., M, N)
        Horizontal gradient(s).

    Returns
    -------
    tuple[NDArray[floating] (..., 2*M, 2*N), NDArray[floating] (..., 2*M, 2*N)]
        Vertical and horizontal gradients with antisymmetric padding.

    """
    xp = array_namespace(gy, gx)
    y, x = xp.broadcast_shapes(gy.shape, gx.shape)[-2:]

    gx_neg = -gx
    gy_neg = -gy

    _sy = slice(y)
    _sx = slice(x)
    sy_ = slice(y, None)
    sx_ = slice(x, None)

    as_gx = xp.empty((*gx.shape[:-2], y * 2, x * 2), dtype=gx.dtype)
    # Equivalent to: as_gx[..., :y, :x] = gx
    as_gx = setitem(as_gx, (..., _sy, _sx), gx)
    # Equivalent to: as_gx[..., y:, :x] = gx[..., ::-1, :]
    as_gx = setitem(as_gx, (..., sy_, _sx), flip(gx, axis=-2))
    # Equivalent to: as_gx[..., :y, x:] = gx_neg[..., ::-1]
    as_gx = setitem(as_gx, (..., _sy, sx_), flip(gx_neg, axis=-1))
    # Equivalent to: as_gx[..., y:, x:] = gx_neg[..., ::-1, ::-1]
    as_gx = setitem(as_gx, (..., sy_, sx_), flip(gx_neg, axis=(-2, -1)))

    as_gy = xp.empty((*gy.shape[:-2], y * 2, x * 2), dtype=gy.dtype)
    # Equivalent to: as_gy[..., :y, :x] = gy
    as_gy = setitem(as_gy, (..., _sy, _sx), gy)
    # Equivalent to: as_gy[..., y:, :x] = gy_neg[..., ::-1, :]
    as_gy = setitem(as_gy, (..., sy_, _sx), flip(gy_neg, axis=-2))
    # Equivalent to: as_gy[..., :y, x:] = gy[..., ::-1]
    as_gy = setitem(as_gy, (..., _sy, sx_), flip(gy, axis=-1))
    # Equivalent to: as_gy[..., y:, x:] = gy_neg[..., ::-1, ::-1]
    as_gy = setitem(as_gy, (..., sy_, sx_), flip(gy_neg, axis=(-2, -1)))

    as_gy, as_gx = as_gx, as_gy

    return as_gy, as_gx


def _antireflect(
    gy: NDArray[floating],
    gx: NDArray[floating],
) -> tuple[NDArray[floating], NDArray[floating]]:
    """Antireflect padding of the input gradients.

    !!! warning "Not in the public API - use at your own risk!"

    !!! warning "Gives worse results than \
        [antisymmetric][mbipy.normal_integration.padding.antisymmetric]."

    Parameters
    ----------
    gy : NDArray[floating] (..., M, N)
        Vertical gradient(s).
    gx : NDArray[floating] (..., M, N)
        Horizontal gradient(s).

    Returns
    -------
    tuple[NDArray[floating] (..., 2*M-1, 2*N-1), NDArray[floating] (..., 2*M-1, 2*N-1)]
        Vertical and horizontal gradients with antireflect padding.
    """
    # !!! gives worse results than antisymmetric - not currently used
    xp = array_namespace(gy, gx)
    y, x = xp.broadcast_shapes(gy.shape, gx.shape)[-2:]

    gx_neg = -gx
    gy_neg = -gy

    _sy = slice(y)
    _sx = slice(x)
    sy_ = slice(y, None)
    sx_ = slice(x, None)

    ar_gx = xp.empty((*gx.shape[:-2], y * 2 - 1, x * 2 - 1), dtype=gx.dtype)
    # Equivalent to: ar_gx[..., :y, :x] = gx
    ar_gx = setitem(ar_gx, (..., _sy, _sx), gx)
    # Equivalent to: ar_gx[..., y:, :x] = gx[..., -1:1:-1, :]
    ar_gx = setitem(ar_gx, (..., sy_, _sx), flip(gx[..., :-1, :], axis=-2))
    # Equivalent to: ar_gx[..., :y, x:] = gx_neg[..., -1:1:-1]
    ar_gx = setitem(ar_gx, (..., _sy, sx_), flip(gx_neg[..., :-1], axis=-1))
    # Equivalent to: ar_gx[..., y:, x:] = gx_neg[..., -1:1:-1, -1:1:-1]
    ar_gx = setitem(ar_gx, (..., sy_, sx_), flip(gx_neg[..., :-1, :-1], axis=(-2, -1)))

    ar_gy = xp.empty((*gy.shape[:-2], y * 2 - 1, x * 2 - 1), dtype=gy.dtype)
    # Equivalent to: ar_gy[..., :y, :x] = gy
    ar_gy = setitem(ar_gy, (..., _sy, _sx), gy)
    # Equivalent to: ar_gy[..., y:, :x] = gy_neg[..., -1:1:-1, :]
    ar_gy = setitem(ar_gy, (..., sy_, _sx), flip(gy_neg[..., :-1, :], axis=-2))
    # Equivalent to: ar_gy[..., :y, x:] = gy[..., -1:1:-1]
    ar_gy = setitem(ar_gy, (..., _sy, sx_), flip(gy[..., :-1], axis=-1))
    # Equivalent to: ar_gy[..., y:, x:] = gy_neg[..., -1:1:-1, -1:1:-1]
    ar_gy = setitem(ar_gy, (..., sy_, sx_), flip(gy_neg[..., :-1, :-1], axis=(-2, -1)))

    return ar_gy, ar_gx
