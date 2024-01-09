"""_summary_
"""

__all__ = (
    "create_similarity_st",
    "create_similarity_svt",
    "create_find_displacement",
    "create_vectors_st",
    "create_vectors_st_svt",
)


import itertools
import warnings

import numpy as np


def cutoff_warning(*arrays, cutoff=None, axis=-1):
    if cutoff is None:
        return
    for i in arrays:
        if i.shape[axis] < cutoff:
            warnings.warn(
                f"Cutoff is too high to take effect, there are {i.shape[axis]} points."
            )


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


def create_similarity_st(xp, swv):
    # TODO nin17: docstring
    docstring = """"""

    def similarity_st(img1_swv, img2_swv, ss, pcc=False):
        if pcc:
            img1_swv = img1_swv - img1_swv.mean(axis=-1, keepdims=True)

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
    def similarity_svt(img1, img2, m, n, pcc=False):
        # TODO nin17: docstring
        _m, _n = 2 * m + 1, 2 * n + 1
        if pcc:
            img1 = img1 - img1.mean(axis=-3, keepdims=True)
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
        _ts = xp.zeros(img1.ndim - 1, dtype=np.int64)
        # _ss[-2:] = ss
        _ts[-2:] = ts

        # idk = _ss[-2:] // 2
        # img2 = img2[..., idk[0] : -idk[0], idk[1] : -idk[1]]
        img1_shape = xp.array(img1.shape[:-3] + img1.shape[-2:])
        img2_shape = xp.array(img2.shape[:-3] + img2.shape[-2:])
        shape1 = xp.array(img1_shape) - _ts
        shape1[-2:] += 1
        shape2 = xp.array(img2_shape) - _ts
        shape2[-2:] += 1
        # TODO do it with transpose and reshape
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
        det = xp.where(denom.astype(bool), 1.0 / denom, 0.0)

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
        disp_y -= array_padded.shape[-2] // 2 - 1
        disp_x -= array_padded.shape[-1] // 2 - 1

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
