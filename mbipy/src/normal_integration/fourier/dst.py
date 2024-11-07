"""Solve the poisson equation with the DST - Dirichlet boundary conditions.

Queau, Y., Durou, J.-D. & Aujol, J.-F. Normal Integration: A Survey.
http://arxiv.org/abs/1709.05940 (Sept. 2017).

Python implementation of: https://github.com/yqueau/normal_integration/blob/1f69b9f1f35bb79457f6a8af753a5d4978811b11/Toolbox/DST_Poisson.m
"""

from __future__ import annotations

__all__ = ("dst_poisson",)


from typing import TYPE_CHECKING

from mbipy.src.config import __have_scipy__
from mbipy.src.normal_integration.utils import check_shapes, r_
from mbipy.src.utils import array_namespace

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

if __have_scipy__:
    from scipy import fft


def dst_poisson(
    gy: NDArray[floating],
    gx: NDArray[floating],
    ub: NDArray[floating] | None = None,
    workers: int | None = None,
) -> NDArray[floating]:
    # !!! Only compatible with NumPy/SciPy & Numba as other libraries lack type-1 DST
    xp = array_namespace(gx, gy)
    dtype = xp.result_type(gy, gx)
    if not xp.isdtype(dtype, "real floating"):
        msg = "Input arrays must be real-valued."
        raise ValueError(msg)
    sy, sx = check_shapes(gx, gy)
    if ub is not None:
        result_shape = xp.broadcast_shapes(gy.shape, gx.shape, ub.shape)
    else:
        result_shape = xp.broadcast_shapes(gy.shape, gx.shape)
    qy = 0.5 * (gy[..., r_((1, sy), -1, xp), :] - gy[..., r_(0, (0, sy - 1), xp), :])
    px = 0.5 * (gx[..., :, r_((1, sx), -1, xp)] - gx[..., :, r_(0, (0, sx - 1), xp)])

    f = qy + px

    if ub is not None:
        # Modification near the boundaries (Eq. 46 in [1])
        f[..., 1, 2:-2] -= ub[..., 0, 2:-2]
        f[..., -2, 2:-2] -= ub[..., -1, 2:-2]
        f[..., 2:-2, 1] -= ub[..., 2:-2, 0]
        f[..., 2:-2, -2] -= ub[..., 2:-2, -1]

        # Modification near the corners (Eq. 47 in [1])
        f[..., 1, 1] -= ub[..., 1, 0] + ub[..., 0, 1]
        f[..., 1, -2] -= ub[..., 1, -1] + ub[..., 0, -2]
        f[..., -2, -2] -= ub[..., -2, -1] + ub[..., -1, -2]
        f[..., -2, 1] -= ub[..., -2, 0] + ub[..., -1, 1]

    fsin = fft.dstn(f[..., 1:-1, 1:-1], axes=(-2, -1), type=1, workers=workers)

    # dtype not supported in numba
    x = xp.astype(xp.linspace(0., xp.pi / 2, sx), dtype, copy=False)[1:-1]
    y = xp.astype(xp.linspace(0., xp.pi / 2, sy), dtype, copy=False)[1:-1][:, None]
    # Faster to do * before + : x.size + y.size vs x.size * y.size
    denom = (-4.0 * xp.sin(x) ** 2) + (-4.0 * xp.sin(y) ** 2)
    z_bar = fsin / denom
    z = xp.zeros(result_shape, dtype=fsin.dtype)
    if ub is not None:
        z[:] = ub
    z[..., 1:-1, 1:-1] = fft.idstn(z_bar, axes=(-2, -1), type=1, workers=workers)
    return z


# def create_dst_poisson(xp, fft):
#     """

#     Parameters
#     ----------
#     xp : _type_
#         _description_
#     fft : _type_
#         _description_

#     Returns
#     -------
#     _type_
#         _description_
#     """

#     if xp.__name__ == "jax.numpy":

#         # TODO(nin17): implement dstn and idstn in JAX

#         def dst_poisson(*, gx, gy, ub=None, **kwargs):
#             sy, sx = check_shapes(gx, gy)
#             if ub is not None:
#                 result_shape = xp.broadcast_shapes(gy.shape, gx.shape, ub.shape)
#             else:
#                 result_shape = xp.broadcast_shapes(gy.shape, gx.shape)
#             if "type" not in kwargs:
#                 kwargs["type"] = 1

#             qy = 0.5 * (gy[..., xp.r_[1:sy, -1], :] - gy[..., xp.r_[0, : sy - 1], :])
#             px = 0.5 * (gx[..., :, xp.r_[1:sx, -1]] - gx[..., :, xp.r_[0, : sx - 1]])
#             f = qy + px

#             if ub is not None:
#                 # Modification near the boundaries (Eq. 46 in [1])
#                 f = f.at[..., 1, 2:-2].add(-ub[..., 0, 2:-2])
#                 f = f.at[..., -2, 2:-2].add(-ub[..., -1, 2:-2])
#                 f = f.at[..., 2:-2, 1].add(-ub[..., 2:-2, 0])
#                 f = f.at[..., 2:-2, -2].add(-ub[..., 2:-2, -1])

#                 # Modification near the corners (Eq. 47 in [1])
#                 f = f.at[..., 1, 1].add(-(ub[..., 1, 0] + ub[..., 0, 1]))
#                 f = f.at[..., 1, -2].add(-(ub[..., 1, -1] + ub[..., 0, -2]))
#                 f = f.at[..., -2, -2].add(-(ub[..., -2, -1] + ub[..., -1, -2]))
#                 f = f.at[..., -2, 1].add(-(ub[..., -2, 0] + ub[..., -1, 1]))

#             fsin = fft.dstn(f[..., 1:-1, 1:-1], axes=(-2, -1), **kwargs)

#             x = xp.linspace(0, xp.pi / 2, sx, dtype=fsin.dtype)[1:-1]
#             y = xp.linspace(0, xp.pi / 2, sy, dtype=fsin.dtype)[1:-1][:, None]
#             # Faster to do * before + : x.size + y.size vs x.size * y.size
#             denom = (-4.0 * xp.sin(x) ** 2) - (4.0 * xp.sin(y) ** 2)
#             z_bar = fsin / denom

#             z = xp.zeros(result_shape, dtype=fsin.dtype)
#             if ub is not None:
#                 z = z.at[:].set(ub)
#             z = z.at[..., 1:-1, 1:-1].set(fft.idstn(z_bar, axes=(-2, -1), **kwargs))
#             return z

#     else:

#         def dst_poisson(*, gx, gy, ub=None, **kwargs):
#             sy, sx = check_shapes(gx, gy)
#             if ub is not None:
#                 result_shape = xp.broadcast_shapes(gy.shape, gx.shape, ub.shape)
#             else:
#                 result_shape = xp.broadcast_shapes(gy.shape, gx.shape)
#             if "type" not in kwargs:
#                 kwargs["type"] = 1

#             qy = 0.5 * (gy[..., xp.r_[1:sy, -1], :] - gy[..., xp.r_[0, : sy - 1], :])
#             px = 0.5 * (gx[..., :, xp.r_[1:sx, -1]] - gx[..., :, xp.r_[0, : sx - 1]])
#             f = qy + px

#             if ub is not None:
#                 # Modification near the boundaries (Eq. 46 in [1])
#                 f[..., 1, 2:-2] -= ub[..., 0, 2:-2]
#                 f[..., -2, 2:-2] -= ub[..., -1, 2:-2]
#                 f[..., 2:-2, 1] -= ub[..., 2:-2, 0]
#                 f[..., 2:-2, -2] -= ub[..., 2:-2, -1]

#                 # Modification near the corners (Eq. 47 in [1])
#                 f[..., 1, 1] -= ub[..., 1, 0] + ub[..., 0, 1]
#                 f[..., 1, -2] -= ub[..., 1, -1] + ub[..., 0, -2]
#                 f[..., -2, -2] -= ub[..., -2, -1] + ub[..., -1, -2]
#                 f[..., -2, 1] -= ub[..., -2, 0] + ub[..., -1, 1]

#             fsin = fft.dstn(f[..., 1:-1, 1:-1], axes=(-2, -1), **kwargs)

#             x = xp.linspace(0, xp.pi / 2, sx, dtype=fsin.dtype)[1:-1]
#             y = xp.linspace(0, xp.pi / 2, sy, dtype=fsin.dtype)[1:-1][:, None]
#             # Faster to do * before + : x.size + y.size vs x.size * y.size
#             denom = (-4.0 * xp.sin(x) ** 2) + (-4.0 * xp.sin(y) ** 2)
#             z_bar = fsin / denom
#             z = xp.zeros(result_shape, dtype=fsin.dtype)
#             if ub is not None:
#                 z[:] = ub
#             z[..., 1:-1, 1:-1] = fft.idstn(z_bar, axes=(-2, -1), **kwargs)
#             return z

#     return dst_poisson
#     return dst_poisson
#     return dst_poisson
