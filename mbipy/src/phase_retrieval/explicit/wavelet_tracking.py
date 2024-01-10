"""
Array based implementations of: WST, WSVT & WST-WSVT.
Compatible with numpy, cupy & jax.
"""

__all__ = "create_wst", "create_wst_wsvt", "create_wsvt"


import types
import typing
import warnings

from .utils import cutoff_warning

# !!! nin17: Don't change WAVEDEC_MODE - only mode="zero" gives correct results
WAVEDEC_MODE = "zero"
PAD_MODE = "reflect"


def wavelet_kwargs(kwargs):
    _wavelet_kwargs = {"wavelet": "db2", "level": None}
    if kwargs is not None:
        _wavelet_kwargs.update(kwargs)
    return _wavelet_kwargs


def level_cutoff_warning(*tuples, level_cutoff=None):
    if level_cutoff is None:
        return
    for i in tuples:
        if len(i) < level_cutoff:
            warnings.warn(
                f"Level cutoff is too high to take effect, there are {len(i)} levels."
            )


def create_wst(
    xp: types.ModuleType,
    vectors_st: typing.Callable,
    similarity_st: typing.Callable,
    find_displacement: typing.Callable,
    wavedec: typing.Callable,
) -> typing.Callable:
    def wst(
        img1,
        img2,
        ss,
        ts,
        wavedec_kwargs=None,
        level_cutoff=None,
        cutoff=None,
    ):
        pcc = False  # nin17: pcc is not preserved in dwt
        _wavelet_kwargs = wavelet_kwargs(wavedec_kwargs)

        vec1, vec2 = vectors_st(img1, img2, ss, ts)

        vec1 = wavedec(vec1, mode=WAVEDEC_MODE, axis=-1, **_wavelet_kwargs)
        vec2 = wavedec(vec2, mode=WAVEDEC_MODE, axis=-1, **_wavelet_kwargs)

        level_cutoff_warning(vec1, vec2, level_cutoff=level_cutoff)
        vec1 = xp.concatenate(vec1[:level_cutoff], -1)
        vec2 = xp.concatenate(vec2[:level_cutoff], -1)

        cutoff_warning(vec1, vec2, cutoff=cutoff, axis=-1)
        vec1 = vec1[..., :cutoff]
        vec2 = vec2[..., :cutoff]

        similarity = similarity_st(vec1, vec2, ss, pcc)
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
        )
        return find_displacement(similarity_padded)

    return wst


def create_wsvt(
    xp: types.ModuleType,
    similarity_svt: typing.Callable,
    find_displacement: typing.Callable,
    wavedec: typing.Callable,
):
    def wsvt(
        img1,
        img2,
        m,
        n,
        wavedec_kwargs=None,
        level_cutoff=None,
        cutoff=None,
    ):
        pcc = False  # nin17: pcc is not preserved in dwt
        _wavelet_kwargs = wavelet_kwargs(wavedec_kwargs)

        vec1 = wavedec(img1, mode=WAVEDEC_MODE, axis=-3, **_wavelet_kwargs)
        vec2 = wavedec(
            img2[..., m:-m, n:-n], mode=WAVEDEC_MODE, axis=-3, **_wavelet_kwargs
        )

        level_cutoff_warning(vec1, vec2, level_cutoff=level_cutoff)
        img1_wt = xp.concatenate(vec1[:level_cutoff], -3)
        img2_wt = xp.concatenate(vec2[:level_cutoff], -3)

        cutoff_warning(img1_wt, img2_wt, cutoff=cutoff, axis=-3)
        img1_wt = img1_wt[..., :cutoff, :, :]
        img2_wt = img2_wt[..., :cutoff, :, :]

        cs = similarity_svt(img1_wt, img2_wt, m, n, pcc)

        cs_padded = xp.pad(cs, ((0, 0),) * (cs.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE)

        return find_displacement(cs_padded)

    return wsvt


def create_wst_wsvt(xp, vectors_st_svt, similarity_st, find_displacement, wavedec):
    def wst_wsvt(
        img1,
        img2,
        ss,
        ts,
        wavedec_kwargs=None,
        level_cutoff=None,
        cutoff=None,
    ):
        pcc = False  # nin17: pcc is not preserved in dwt
        _wavelet_kwargs = wavelet_kwargs(wavedec_kwargs)

        vec1, vec2 = vectors_st_svt(img1, img2, ss, ts)

        vec1 = wavedec(vec1, mode=WAVEDEC_MODE, axis=-1, **_wavelet_kwargs)
        vec2 = wavedec(vec2, mode=WAVEDEC_MODE, axis=-1, **_wavelet_kwargs)

        level_cutoff_warning(vec1, vec2, level_cutoff=level_cutoff)
        vec1 = xp.concatenate(vec1[:level_cutoff], -1)
        vec2 = xp.concatenate(vec2[:level_cutoff], -1)

        cutoff_warning(vec1, vec2, cutoff=cutoff, axis=-1)
        vec1 = vec1[..., :cutoff]
        vec2 = vec2[..., :cutoff]

        similarity = similarity_st(vec1, vec2, ss, pcc)
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
        )
        return find_displacement(similarity_padded)

    return wst_wsvt
