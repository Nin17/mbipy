"""_summary_
"""

from __future__ import annotations

__all__ = ("antisym",)

from typing import TYPE_CHECKING

from mbipy.src.config import __have_numba__
from mbipy.src.normal_integration.utils import check_shapes
from mbipy.src.utils import array_namespace, setitem

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


def antisym(
    gy: NDArray[floating], gx: NDArray[floating]
) -> tuple[NDArray[floating], NDArray[floating]]:
    xp = array_namespace(gy, gx)

    gx_neg = -gx
    gy_neg = -gy

    y, x = check_shapes(gy, gx)

    _sy = slice(y)
    _sx = slice(x)
    sy_ = slice(y, None)
    sx_ = slice(x, None)

    as_gx = xp.empty(gx.shape[:-2] + (y * 2, x * 2), dtype=gx.dtype)
    # as_gx[..., :y, :x] = gx
    as_gx = setitem(as_gx, (..., _sy, _sx), gx, xp)
    # as_gx[..., y:, :x] = gx[..., ::-1, :]
    as_gx = setitem(as_gx, (..., sy_, _sx), xp.flip(gx, axis=-2), xp)
    # as_gx[..., :y, x:] = gx_neg[..., ::-1]
    as_gx = setitem(as_gx, (..., _sy, sx_), xp.flip(gx_neg, axis=-1), xp)
    # as_gx[..., y:, x:] = gx_neg[..., ::-1, ::-1]
    as_gx = setitem(as_gx, (..., sy_, sx_), xp.flip(gx_neg, axis=(-2, -1)), xp)

    as_gy = xp.empty(gy.shape[:-2] + (y * 2, x * 2), dtype=gy.dtype)
    # as_gy[..., :y, :x] = gy
    as_gy = setitem(as_gy, (..., _sy, _sx), gy, xp)
    # as_gy[..., y:, :x] = gy_neg[..., ::-1, :]
    as_gy = setitem(as_gy, (..., sy_, _sx), xp.flip(gy_neg, axis=-2), xp)
    # as_gy[..., :y, x:] = gy[..., ::-1]
    as_gy = setitem(as_gy, (..., _sy, sx_), xp.flip(gy, axis=-1), xp)
    # as_gy[..., y:, x:] = gy_neg[..., ::-1, ::-1]
    as_gy = setitem(as_gy, (..., sy_, sx_), xp.flip(gy_neg, axis=(-2, -1)), xp)

    return as_gy, as_gx


if __have_numba__:
    import numba as nb
    from numba import types
    from numpy import flip

    nb.extending.register_jitable(antisym)

    @nb.extending.overload(flip)
    def flip_overload(a, axis=None):
        if isinstance(axis, types.Integer):

            def impl(a, axis=None):
                if axis == -1:
                    return a[..., ::-1]
                if axis == -2:
                    return a[..., ::-1, :]
                raise ValueError("Invalid axis.")

        elif isinstance(axis, types.UniTuple):

            def impl(a, axis=None):
                if axis == (-2, -1):
                    return a[..., ::-1, ::-1]
                raise ValueError("Invalid axis")

        else:
            raise ValueError("Invalid axis type.")
        return impl


# def create_antisym(xp: types.ModuleType) -> typing.Callable:
#     """
#     Decorator to pad the vertical and horizonatl components of the normal
#     vector field antisymmetrically.

#     Parameters
#     ----------
#     func : typing.Callable
#         Normal integration function that requires the inputs to be
#         antisymmetrically padded

#     Returns
#     -------
#     typing.Callable
#         Wrapped function with the input padded and the output sliced to the
#         original shape
#     """

#     if hasattr(xp, "block"):

#         def antisym(*, gy, gx):
#             # assert gx.shape[-2:] == gy.shape[-2:]
#             # assert gx.ndim >= 2
#             gx_neg = -gx
#             gy_neg = -gy
#             antisym_gx = xp.block(
#                 [
#                     [gx, gx_neg[..., ::-1]],
#                     [gx[..., ::-1, :], gx_neg[..., ::-1, ::-1]],
#                 ]
#             )
#             antisym_gy = xp.block(
#                 [
#                     [gy, gy[..., ::-1]],
#                     [gy_neg[..., ::-1, :], gy_neg[..., ::-1, ::-1]],
#                 ]
#             )
#             return antisym_gy, antisym_gx

#     else:

#         def antisym(*, gy, gx):
#             # assert gx.shape[-2:] == gy.shape[-2:]
#             # assert gx.ndim >= 2
#             # y, x = gx.shape[-2:]
#             y, x = xp.broadcast_shapes(gy.shape[-2:], gx.shape[-2:])
#             gx_neg = -gx
#             gy_neg = -gy
#             antisym_gx = xp.empty(gx.shape[:-2] + (y * 2, x * 2), dtype=gx.dtype)
#             antisym_gx[..., :y, :x] = gx
#             antisym_gx[..., y:, :x] = gx[..., ::-1, :]
#             antisym_gx[..., :y, x:] = gx_neg[..., ::-1]
#             antisym_gx[..., y:, x:] = gx_neg[..., ::-1, ::-1]

#             antisym_gy = xp.empty(gy.shape[:-2] + (y * 2, x * 2), dtype=gy.dtype)
#             antisym_gy[..., :y, :x] = gy
#             antisym_gy[..., y:, :x] = gy_neg[..., ::-1, :]
#             antisym_gy[..., :y, x:] = gy[..., ::-1]
#             antisym_gy[..., y:, x:] = gy_neg[..., ::-1, ::-1]

#             return antisym_gy, antisym_gx

#     return antisym


# def create_antisym_numba():
#     nb = importlib.import_module("numba")
#     np = importlib.import_module("numpy")

#     @nb.extending.register_jitable
#     def antisym_numba(gy, gx):
#         with nb.objmode(x="int64", y="int64"):
#             y, x = np.broadcast_shapes(gy.shape[-2:], gx.shape[-2:])
#         gx_neg = -gx
#         gy_neg = -gy
#         antisym_gx = np.empty(gx.shape[:-2] + (y * 2, x * 2), dtype=gx.dtype)
#         antisym_gx[..., :y, :x] = gx
#         antisym_gx[..., y:, :x] = gx[..., ::-1, :]
#         antisym_gx[..., :y, x:] = gx_neg[..., ::-1]
#         antisym_gx[..., y:, x:] = gx_neg[..., ::-1, ::-1]

#         antisym_gy = np.empty(gy.shape[:-2] + (y * 2, x * 2), dtype=gy.dtype)
#         antisym_gy[..., :y, :x] = gy
#         antisym_gy[..., y:, :x] = gy_neg[..., ::-1, :]
#         antisym_gy[..., :y, x:] = gy[..., ::-1]
#         antisym_gy[..., y:, x:] = gy_neg[..., ::-1, ::-1]

#         return antisym_gy, antisym_gx

#     return antisym_numba
#     return antisym_numba


# # if hasattr(xp, "block"):
# #     gx_neg = -gx
# #     gy_neg = -gy

# #     antisym_gx = xp.block(
# #         [
# #             [gx, gx_neg[..., ::-1]],
# #             [gx[..., ::-1, :], gx_neg[..., ::-1, ::-1]],
# #         ]
# #     )
# #     antisym_gy = xp.block(
# #         [
# #             [gy, gy[..., ::-1]],
# #             [gy_neg[..., ::-1, :], gy_neg[..., ::-1, ::-1]],
# #         ]
# #     )
# #     return antisym_gy, antisym_gx
# #     return antisym_gy, antisym_gx
