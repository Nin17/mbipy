"""Solve the poisson equation with the DCT - homogeneous Neumann boundary conditions.

Queau, Y., Durou, J.-D. & Aujol, J.-F. Normal Integration: A Survey.
http://arxiv.org/abs/1709.05940 (Sept. 2017).

Python implementation of: https://github.com/yqueau/normal_integration/blob/1f69b9f1f35bb79457f6a8af753a5d4978811b11/Toolbox/DCT_Poisson.m
"""

from __future__ import annotations

__all__ = ("dct_poisson",)


from typing import TYPE_CHECKING

from mbipy.src.normal_integration.utils import check_shapes, get_dcts, r_
from mbipy.src.utils import array_namespace, isub, setitem

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


def dct_poisson(
    gy: NDArray[floating], gx: NDArray[floating], workers: int | None = None
) -> NDArray[floating]:
    xp = array_namespace(gy, gx)
    dtype = xp.result_type(gy, gx)
    if not xp.isdtype(dtype, "real floating"):
        msg = "Input arrays must be real-valued."
        raise ValueError(msg)
    dctn, idctn = get_dcts(xp)

    sy, sx = check_shapes(gx, gy)

    # Divergence of (gy, gx) using central differences
    qy = 0.5 * (gy[..., r_((1, sy), -1, xp), :] - gy[..., r_(0, (0, sy - 1), xp), :])
    px = 0.5 * (gx[..., :, r_((1, sx), -1, xp)] - gx[..., :, r_(0, (0, sx - 1), xp)])

    # Div(gy, gx)
    f = qy + px

    # Modification near the boundaries to enforce the non-homogeneous Neumann
    # BC (Eq. 53 in [1])
    s1_1 = slice(1, -1)
    # f[..., 0, 1:-1] -= gy[..., 0, 1:-1]
    f = isub(f, (..., 0, s1_1), gy[..., 0, s1_1], xp)
    # f[..., -1, 1:-1] -= gy[..., -1, 1:-1]
    f = isub(f, (..., -1, s1_1), gy[..., -1, s1_1], xp)
    # f[..., 1:-1, 0] -= gx[..., 1:-1, 0]
    f = isub(f, (..., s1_1, 0), gx[..., s1_1, 0], xp)
    # f[..., 1:-1, -1] -= gx[..., 1:-1, -1]
    f = isub(f, (..., s1_1, -1), gx[..., s1_1, -1], xp)

    # Modification near the corners (Eq. 54 in [1])
    # f[..., 0, -1] -= -gy[..., 0, 0] - gx[..., 0, 0]
    f = isub(f, (..., 0, -1), -gy[..., 0, 0] - gx[..., 0, 0], xp)
    # f[..., -1, -1] -= gy[..., -1, -1] + gx[..., -1, -1]
    f = isub(f, (..., -1, -1), gy[..., -1, -1] + gx[..., -1, -1], xp)
    # f[..., -1, 0] -= gy[..., -1, 0] - gx[..., -1, 0]
    f = isub(f, (..., -1, 0), gy[..., -1, 0] - gx[..., -1, 0], xp)
    # f[..., 0, 0] -= -gy[..., 0, -1] + gx[..., 0, -1]
    f = isub(f, (..., 0, 0), -gy[..., 0, -1] + gx[..., 0, -1], xp)

    fcos = dctn(f, axes=(-2, -1), workers=workers)

    # dtype not supported in numba
    x = xp.astype(xp.linspace(0, xp.pi / 2, sx), dtype, copy=False)
    y = xp.astype(xp.linspace(0, xp.pi / 2, sy), dtype, copy=False)[:, None]
    # Faster to do * before + : x.size + y.size vs x.size * y.size
    denom = 4.0 * xp.sin(x) ** 2 + 4.0 * xp.sin(y) ** 2
    denom = setitem(denom, (0, 0), 1.0, xp)
    z_bar_bar = -fcos / denom

    return idctn(z_bar_bar, axes=(-2, -1), workers=workers)


# def create_dct_poisson(xp, fft):
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

#     Raises
#     ------
#     ValueError
#         _description_
#     ValueError
#         _description_
#     """

#     if xp.__name__ == "jax.numpy":

#         def dct_poisson(*, gx, gy, **kwargs):
#             sy, sx = check_shapes(gx, gy)

#             # Divergence of (gy, gx) using central differences
#             qy = 0.5 * (gy[..., xp.r_[1:sy, -1], :] - gy[..., xp.r_[0, : sy - 1], :])
#             px = 0.5 * (gx[..., :, xp.r_[1:sx, -1]] - gx[..., :, xp.r_[0, : sx - 1]])

#             # Div(gy, gx)
#             f = qy + px

#             # Modification near the boundaries to enforce the non-homogeneous Neumann
#             # BC (Eq. 53 in [1])
#             f = f.at[..., 0, 1:-1].add(-gy[..., 0, 1:-1])
#             f = f.at[..., -1, 1:-1].add(-gy[..., -1, 1:-1])
#             f = f.at[..., 1:-1, 0].add(-gx[..., 1:-1, 0])
#             f = f.at[..., 1:-1, -1].add(-gx[..., 1:-1, -1])

#             # Modification near the corners (Eq. 54 in [1])
#             f = f.at[..., 0, -1].add(-(-gy[..., 0, 0] - gx[..., 0, 0]))
#             f = f.at[..., -1, -1].add(-(gy[..., -1, -1] + gx[..., -1, -1]))
#             f = f.at[..., -1, 0].add(-(gy[..., -1, 0] - gx[..., -1, 0]))
#             f = f.at[..., 0, 0].add(-(-gy[..., 0, -1] + gx[..., 0, -1]))

#             fcos = fft.dctn(f, axes=(-2, -1), **kwargs)

#             x = xp.linspace(0, xp.pi / 2, sx, dtype=fcos.dtype)
#             y = xp.linspace(0, xp.pi / 2, sy, dtype=fcos.dtype)[:, None]
#             # Faster to do * before + : x.size + y.size vs x.size * y.size
#             denom = 4.0 * xp.sin(x) ** 2 + 4.0 * xp.sin(y) ** 2
#             # denom = denom.at[0, 0].set(xp.finfo(fcos.dtype).eps)
#             z_bar_bar = -fcos / denom
#             z_bar_bar = z_bar_bar.at[..., 0, 0].set(fcos[..., 0, 0])

#             z = fft.idctn(z_bar_bar, axes=(-2, -1), **kwargs)

#             return z

#     else:

#         def dct_poisson(*, gx, gy, **kwargs):
#             sy, sx = check_shapes(gx, gy)
#             # Divergence of (gy, gx) using central differences
#             qy = 0.5 * (gy[..., xp.r_[1:sy, -1], :] - gy[..., xp.r_[0, : sy - 1], :])
#             px = 0.5 * (gx[..., :, xp.r_[1:sx, -1]] - gx[..., :, xp.r_[0, : sx - 1]])

#             # Div(gy, gx)
#             f = qy + px

#             # Modification near the boundaries to enforce the non-homogeneous Neumann
#             # BC (Eq. 53 in [1])
#             f[..., 0, 1:-1] -= gy[..., 0, 1:-1]
#             f[..., -1, 1:-1] -= gy[..., -1, 1:-1]
#             f[..., 1:-1, 0] -= gx[..., 1:-1, 0]
#             f[..., 1:-1, -1] -= gx[..., 1:-1, -1]

#             # Modification near the corners (Eq. 54 in [1])
#             f[..., 0, -1] -= -gy[..., 0, 0] - gx[..., 0, 0]
#             f[..., -1, -1] -= gy[..., -1, -1] + gx[..., -1, -1]
#             f[..., -1, 0] -= gy[..., -1, 0] - gx[..., -1, 0]
#             f[..., 0, 0] -= -gy[..., 0, -1] + gx[..., 0, -1]

#             fcos = fft.dctn(f, axes=(-2, -1), **kwargs)

#             x = xp.linspace(0, xp.pi / 2, sx, dtype=fcos.dtype)
#             y = xp.linspace(0, xp.pi / 2, sy, dtype=fcos.dtype)[:, None]
#             # Faster to do * before + : x.size + y.size vs x.size * y.size
#             denom = 4.0 * xp.sin(x) ** 2 + 4.0 * xp.sin(y) ** 2
#             denom[0, 0] = xp.finfo(fcos.dtype).eps
#             z_bar_bar = -fcos / denom
#             z_bar_bar[..., 0, 0] = fcos[..., 0, 0]

#             z = fft.idctn(z_bar_bar, axes=(-2, -1), **kwargs)

#             return z

#     return dct_poisson
#     return dct_poisson
