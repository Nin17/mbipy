"""_summary_
"""

__all__ = (
    "create_find_displacement",
    "create_find_displacement2",
    "create_similarity_st",
    "create_similarity_svt",
    "create_vectors_st",
    "create_vectors_st_svt",
)

import importlib
import itertools
import warnings

import numpy as np
from array_api_compat import is_cupy_namespace, is_numpy_namespace


def get_correlate1d(xp):
    if is_numpy_namespace(xp):
        return importlib.import_module("scipy.ndimage").correlate1d
    if is_cupy_namespace(xp):
        return importlib.import_module("cupyx.scipy.ndimage").correlate1d
    raise NotImplementedError(f"{xp.__name__} not supported")

def get_swv(xp):
    if is_numpy_namespace(xp):
        return importlib.import_module("numpy.lib.stride_tricks").sliding_window_view
    if is_cupy_namespace(xp):
        return importlib.import_module("cupy.lib.stride_tricks").sliding_window_view
    # if hasattr(xp, "lib.stride_tricks"):
    #     return xp.lib.stride_tricks.sliding_window_view
    raise NotImplementedError(f"{xp.__name__} not supported")

def assert_odd(*args: tuple[int, ...]) -> None:
    if not all(i % 2 == 1 for i in args):
        raise ValueError("All search and template dimensions must be odd.")



def cutoff_warning(*arrays, cutoff=None, axis=-1):
    if cutoff is None:
        return
    for i in arrays:
        if i.shape[axis] < cutoff:
            warnings.warn(
                f"Cutoff is too high to take effect, there are {i.shape[axis]} points.",
            )


def create_similarity_st(xp, swv):
    # TODO nin17: docstring
    docstring = """"""

    def similarity_st(img1_swv, img2_swv, ss):

        img1_swv_norm = xp.linalg.norm(img1_swv, axis=-1)

        img1_swv = img1_swv / img1_swv_norm[..., None]

        img1_swv_swv = swv(img1_swv, ss, axis=(-3, -2))
        return xp.einsum(
            "...ijklm, ...ijklm -> ...ijlm",
            img1_swv_swv,
            img2_swv[..., None, None],
            optimize="optimal",
        )

    similarity_st.__doc__ = docstring
    return similarity_st


def create_similarity_svt(xp, swv):
    def similarity_svt(img1, img2, m, n):
        # TODO nin17: docstring
        _m, _n = 2 * m + 1, 2 * n + 1
        norm1 = xp.linalg.norm(img1, axis=-3, keepdims=True)
        img1 = img1 / norm1
        img1_swv = swv(img1, (_m, _n), axis=(-2, -1))
        return xp.einsum(
            "...ijklm, ...ijklm -> ...jklm",
            img1_swv,
            img2[..., None, None],
            optimize="optimal",
        )

    return similarity_svt


def create_vectors_st(swv):
    def vectors_st(img1, img2, ss, ts):
        assert img1.ndim == img2.ndim
        assert img1.ndim >= 2
        if not all(i % 2 == 1 for i in itertools.chain(ss, ts)):
            raise ValueError("All search and template dimensions must be odd.")

        m1, m2, n1, n2 = None, None, None, None
        if ss[0] > 1:
            m1 = ss[0] // 2
            m2 = -m1
        if ss[1] > 1:
            n1 = ss[1] // 2
            n2 = -n1

        img2 = img2[..., m1:m2, n1:n2]

        _ts = np.zeros(img1.ndim, dtype=np.int64)
        _ts[-2:] = ts
        shape1 = np.array(img1.shape) - _ts
        shape1[-2:] += 1
        shape2 = np.array(img2.shape) - _ts
        shape2[-2:] += 1

        img1_swv = swv(img1, ts, axis=(-2, -1)).reshape(*shape1, -1)
        img2_swv = swv(img2, ts, axis=(-2, -1)).reshape(*shape2, -1)
        return img1_swv, img2_swv

    return vectors_st


def create_vectors_st_svt(xp, swv):
    def vectors_st_svt(img1, img2, ss, ts):
        assert img1.ndim == img2.ndim
        assert img1.ndim >= 3
        if not all(i % 2 == 1 for i in itertools.chain(ss, ts)):
            raise ValueError("All search and template dimensions must be odd.")

        m1, m2, n1, n2 = None, None, None, None
        if ss[0] > 1:
            m1 = ss[0] // 2
            m2 = -m1
        if ss[1] > 1:
            n1 = ss[1] // 2
            n2 = -n1

        img2 = img2[..., m1:m2, n1:n2]

        # _ss = xp.zeros(img1.ndim - 1, dtype=np.int64)
        _ts = np.zeros(img1.ndim - 1, dtype=np.int64)
        # _ss[-2:] = ss
        _ts[-2:] = ts

        # idk = _ss[-2:] // 2
        # img2 = img2[..., idk[0] : -idk[0], idk[1] : -idk[1]]
        img1_shape = np.array(img1.shape[:-3] + img1.shape[-2:])
        img2_shape = np.array(img2.shape[:-3] + img2.shape[-2:])
        shape1 = np.array(img1_shape) - _ts
        shape1[-2:] += 1
        shape2 = np.array(img2_shape) - _ts
        shape2[-2:] += 1
        # TODO do it with transpose and reshape
        ts = tuple(ts)
        shape1 = tuple(shape1)
        shape2 = tuple(shape2)
        img1_swv = swv(img1, ts, axis=(-2, -1))
        img1_swv = xp.moveaxis(img1_swv, -5, -3).reshape(*shape1, -1)
        img2_swv = swv(img2, ts, axis=(-2, -1))
        img2_swv = xp.moveaxis(img2_swv, -5, -3).reshape(*shape2, -1)
        return img1_swv, img2_swv

    return vectors_st_svt


def create_find_displacement(xp):
    # TODO nin17: add docstring
    docstring = """_summary_"""

    # FIXME nin17: this has to be wrong - how can it be larger than the window size with
    # padding = reflect

    def find_displacement(array_padded):
        window_shape = tuple(np.asarray(array_padded.shape[-2:]) - 2)
        dy_int, dx_int = xp.unravel_index(
            array_padded[..., 1:-1, 1:-1]
            .reshape(*array_padded.shape[:-2], -1)
            .argmax(-1),
            window_shape,
        )
        ndim = array_padded.ndim - 3
        # TODO nin17: can probably do this with ndarray.take
        preceding = tuple(
            xp.arange(j).reshape((1,) * i + (-1,) + (1,) * (ndim - i))
            for i, j in enumerate(array_padded.shape[:-2])
        )

        dy_int1 = dy_int + 1
        dy_int2 = dy_int + 2

        dx_int1 = dx_int + 1
        dx_int2 = dx_int + 2

        dy = (
            array_padded[preceding + (dy_int2, dx_int1)]
            - array_padded[preceding + (dy_int, dx_int1)]
        ) / 2.0
        dyy = (
            array_padded[preceding + (dy_int2, dx_int1)]
            + array_padded[preceding + (dy_int, dx_int1)]
            - 2.0 * array_padded[preceding + (dy_int1, dx_int1)]
        )
        dx = (
            array_padded[preceding + (dy_int1, dx_int2)]
            - array_padded[preceding + (dy_int1, dx_int)]
        ) / 2.0
        dxx = (
            array_padded[preceding + (dy_int1, dx_int2)]
            + array_padded[preceding + (dy_int1, dx_int)]
            - 2.0 * array_padded[preceding + (dy_int1, dx_int1)]
        )
        dxy = (
            array_padded[preceding + (dy_int2, dx_int2)]
            - array_padded[preceding + (dy_int2, dx_int)]
            - array_padded[preceding + (dy_int, dx_int2)]
            + array_padded[preceding + (dy_int, dx_int)]
        ) / 4.0

        # TODO nin17: sort this out
        denom = dxx * dyy - dxy * dxy
        # det = xp.where(denom, 1.0 / denom, 0.0)
        det = xp.where(denom > np.finfo(np.float64).eps, 1.0 / denom, 0.0)

        # disp_x = -(dyy * dx - dxy * dy) / denom
        # disp_y = -(dxx * dy - dxy * dx) / denom
        disp_x = -(dyy * dx - dxy * dy) * det
        disp_y = -(dxx * dy - dxy * dx) * det

        disp_y += dy_int
        disp_x += dx_int

        # TODO nin17: remove this temporary fix
        # ??? nin17: why -2
        disp_y = xp.clip(disp_y, 0, array_padded.shape[-2] - 1)
        disp_x = xp.clip(disp_x, 0, array_padded.shape[-1] - 1)

        # ??? nin17: why -2
        disp_y = disp_y - array_padded.shape[-2] // 2 - 1
        disp_x = disp_x - array_padded.shape[-1] // 2 - 1

        return disp_y, disp_x

    find_displacement.__doc__ = docstring
    return find_displacement


# TODO nin17: see if this is faster
def nin17(img_pad):
    posy, posx = np.unravel_index(
        img_pad[..., 1:-1, 1:-1].reshape(img_pad.shape[:-2] + (-1,)).argmax(axis=-1),
        np.array(img_pad.shape[-2:]) - 2,
    )

    tup = tuple(range(img_pad.ndim - 2))
    inds = tuple(
        np.expand_dims(np.arange(j), tup[:i] + tup[i + 1 :])
        for i, j in enumerate(img_pad.shape[:-2])
    )

    dy = (img_pad[inds + (posy + 2, posx + 1)] - img_pad[inds + (posy, posx + 1)]) / 2.0
    dyy = (
        img_pad[inds + (posy + 2, posx + 1)]
        + img_pad[inds + (posy, posx + 1)]
        - 2.0 * img_pad[inds + (posy + 1, posx + 1)]
    )

    dx = (img_pad[inds + (posy + 1, posx + 2)] - img_pad[inds + (posy + 1, posx)]) / 2.0
    dxx = (
        img_pad[inds + (posy + 1, posx + 2)]
        + img_pad[inds + (posy + 1, posx)]
        - 2.0 * img_pad[inds + (posy + 1, posx + 1)]
    )

    dxy = (
        img_pad[inds + (posy + 2, posx + 2)]
        - img_pad[inds + (posy + 2, posx)]
        - img_pad[inds + (posy, posx + 2)]
        + img_pad[inds + (posy, posx)]
    ) / 4.0

    denom = dxx * dyy - dxy * dxy
    denom[denom == 0.0] = 1.0

    dispx = -(dyy * dx - dxy * dy) / denom
    dispy = -(dxx * dy - dxy * dx) / denom
    # TODO they're equivalent up to here - decide what to do
    # Their thing seems to be to do with the pyramid scheme

    return posy + np.clip(dispy, -1.0, 1.0), posx + np.clip(dispx, -1.0, 1.0)


def create_find_displacement2(xp):
    def find_disp_np(array):
        sy, sx = np.asarray(array.shape[-2:])
        sy2 = sy // 2
        sx2 = sx // 2
        assert sy % 2 == 1 and sx % 2 == 1
        maxy = sy - 1
        maxx = sx - 1
        dy_int1, dx_int1 = np.unravel_index(
            array.reshape(array.shape[:-2] + (-1,)).argmax(axis=-1), array.shape[-2:],
        )
        # dy_int1, dx_int1 = xp.divmod(
        #     array.reshape(array.shape[:-2] + (-1,)).argmax(axis=-1), sx
        # )
        dy_int1 = dy_int1.ravel()
        dx_int1 = dx_int1.ravel()

        dy_int = xp.where(dx_int1 > 0, dy_int1, 1)
        dx_int = xp.where(dy_int1 > 0, dx_int1, 1)
        # dy_int = xp.abs(dy_int1 - 1)
        # dx_int = xp.abs(dx_int1 - 1)

        dy_int2 = xp.where(dy_int1 < maxy, dy_int1 + 1, maxy - 1)
        dx_int2 = xp.where(dx_int1 < maxx, dx_int1 + 1, maxx - 1)
        # dy_int2 = maxy - xp.abs(dy_int1 + (1 - maxy))
        # dx_int2 = maxx - xp.abs(dx_int1 + (1 - maxx))

        _shape = array.shape[:-2]

        prod = xp.prod(array.shape[:-2])
        prod_array = xp.arange(prod, dtype=np.int64)
        array = array.reshape((-1,) + array.shape[-2:])

        dy = (
            array[prod_array, dy_int2, dx_int1] - array[prod_array, dy_int, dx_int1]
        ) / 2.0
        dyy = (
            array[prod_array, dy_int2, dx_int1]
            - 2 * array[prod_array, dy_int1, dx_int1]
            + array[prod_array, dy_int, dx_int1]
        )

        dx = (
            array[prod_array, dy_int1, dx_int2] - array[prod_array, dy_int1, dx_int]
        ) / 2.0
        dxx = (
            array[prod_array, dy_int1, dx_int2]
            - 2 * array[prod_array, dy_int1, dx_int1]
            + array[prod_array, dy_int1, dx_int]
        )

        dxy = (
            array[prod_array, dy_int2, dx_int2]
            - array[prod_array, dy_int2, dx_int]
            - array[prod_array, dy_int, dx_int2]
            + array[prod_array, dy_int, dx_int]
        ) / 4.0

        denom = dxx * dyy - dxy**2

        det = xp.where(denom > np.finfo(np.float64).eps, 1 / denom, 0)

        disp_x = -(dyy * dx - dxy * dy) * det
        disp_y = -(dxx * dy - dxy * dx) * det

        return xp.clip(
            disp_y.reshape(_shape) + dy_int1.reshape(_shape) - sy2, -sy2, sy2,
        ), xp.clip(disp_x.reshape(_shape) + dx_int1.reshape(_shape) - sx2, -sx2, sx2)

    return find_disp_np


# !!! numba implementation
# @nb.njit(fastmath=True)
# def find_disp(array):
#     # array is 2d window
#     assert array.ndim == 2
#     sy, sx = array.shape
#     argmaximum = array.argmax()
#     dy_int1, dx_int1 = divmod(argmaximum, sx)
#     maxy = sy - 1
#     maxx = sx - 1

#     dy_int = dy_int1 -1 if dy_int1 > 0 else 1
#     dx_int = dx_int1 -1 if dx_int1 > 0 else 1

#     dy_int2 = dy_int1 + 1 if dy_int1 < maxy else maxy - 1
#     dx_int2 = dx_int1 + 1 if dx_int1 < maxx else maxx - 1

#     dy = (array[dy_int2, dx_int1] - array[dy_int, dx_int1]) / 2.0
#     dyy = array[dy_int2, dx_int1] - 2*array[dy_int1, dx_int1] + array[dy_int, dx_int1]

#     dx = (array[dy_int1, dx_int2] - array[dy_int1, dx_int]) / 2.0
#     dxx = array[dy_int1, dx_int2] - 2*array[dy_int1, dx_int1] + array[dy_int1, dx_int]

#     dxy = (array[dy_int2, dx_int2] - array[dy_int2, dx_int] - array[dy_int, dx_int2] + array[dy_int, dx_int]) / 4.0

#     denom = dxx*dyy - dxy**2

#     det = 1 / denom if denom > np.finfo(np.float64).eps else 0

#     disp_x = - (dyy*dx - dxy*dy) * det
#     disp_y = - (dxx*dy - dxy*dx) * det
#     return disp_y + dy_int1 - sy // 2, disp_x + dx_int1 - sx // 2

from ...utils import array_namespace


def find_displacement(array_padded):
    # TODO(nin17): add docstring
    # From create_find_displacement
    xp = array_namespace(array_padded)
    window_shape = tuple(np.asarray(array_padded.shape[-2:]) - 2)
    dy_int, dx_int = xp.unravel_index(
        array_padded[..., 1:-1, 1:-1]
        .reshape(*array_padded.shape[:-2], -1)
        .argmax(-1),
        window_shape,
    )
    ndim = array_padded.ndim - 3
    # TODO nin17: can probably do this with ndarray.take
    preceding = tuple(
        xp.arange(j).reshape((1,) * i + (-1,) + (1,) * (ndim - i))
        for i, j in enumerate(array_padded.shape[:-2])
    )

    dy_int1 = dy_int + 1
    dy_int2 = dy_int + 2

    dx_int1 = dx_int + 1
    dx_int2 = dx_int + 2

    dy = (
        array_padded[preceding + (dy_int2, dx_int1)]
        - array_padded[preceding + (dy_int, dx_int1)]
    ) / 2.0
    dyy = (
        array_padded[preceding + (dy_int2, dx_int1)]
        + array_padded[preceding + (dy_int, dx_int1)]
        - 2.0 * array_padded[preceding + (dy_int1, dx_int1)]
    )
    dx = (
        array_padded[preceding + (dy_int1, dx_int2)]
        - array_padded[preceding + (dy_int1, dx_int)]
    ) / 2.0
    dxx = (
        array_padded[preceding + (dy_int1, dx_int2)]
        + array_padded[preceding + (dy_int1, dx_int)]
        - 2.0 * array_padded[preceding + (dy_int1, dx_int1)]
    )
    dxy = (
        array_padded[preceding + (dy_int2, dx_int2)]
        - array_padded[preceding + (dy_int2, dx_int)]
        - array_padded[preceding + (dy_int, dx_int2)]
        + array_padded[preceding + (dy_int, dx_int)]
    ) / 4.0

    # TODO nin17: sort this out
    denom = dxx * dyy - dxy * dxy
    # det = xp.where(denom, 1.0 / denom, 0.0)
    det = xp.where(denom > np.finfo(np.float64).eps, 1.0 / denom, 0.0)

    # disp_x = -(dyy * dx - dxy * dy) / denom
    # disp_y = -(dxx * dy - dxy * dx) / denom
    disp_x = -(dyy * dx - dxy * dy) * det
    disp_y = -(dxx * dy - dxy * dx) * det

    disp_y += dy_int
    disp_x += dx_int

    # TODO nin17: remove this temporary fix
    # ??? nin17: why -2
    disp_y = xp.clip(disp_y, 0., array_padded.shape[-2] - 1.)
    disp_x = xp.clip(disp_x, 0., array_padded.shape[-1] - 1.)

    # ??? nin17: why -2
    disp_y = disp_y - array_padded.shape[-2] // 2 - 1
    disp_x = disp_x - array_padded.shape[-1] // 2 - 1

    return disp_y, disp_x
